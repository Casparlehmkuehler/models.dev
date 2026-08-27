import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { z } from "zod";

import { ReasoningOption } from "../../schema.js";
import type { ExistingModel, SyncProvider, SyncedBaseModel } from "../index.js";

const API_BASE = "https://api.cloudflare.com/client/v4/accounts";
const PROVIDER_DIR = path.join(
  import.meta.dirname,
  "..",
  "..",
  "..",
  "..",
  "..",
  "providers",
  "cloudflare-ai-gateway",
);
const MODELS_ROOT = path.join(import.meta.dirname, "..", "..", "..", "..", "..", "models");
const CURATION_PATH = path.join(PROVIDER_DIR, "curation.toml");
const TEXT_GENERATION = "Text Generation";
const REQUEST_TIMEOUT_MS = 30_000;

const NATIVE_NPM: Record<string, string> = {
  anthropic: "@ai-sdk/anthropic",
  openai: "@ai-sdk/openai",
};

const CatalogEntry = z.object({
  model_id: z.string().refine(isSafeModelID, "model_id must be a safe relative provider/model path"),
  task: z.string(),
  context_length: z.number().int().positive().nullish(),
  pricing: z.record(z.number().nonnegative()).nullish(),
}).passthrough();
const CatalogModel = CatalogEntry.extend({
  task: z.literal(TEXT_GENERATION),
  context_length: z.number().int().positive().nullish(),
  pricing: z.record(z.number().nonnegative()),
});

const CloudflareResponse = z.object({
  result: z.array(CatalogEntry),
  result_info: z.object({
    total_count: z.number().int().nonnegative(),
  }).passthrough(),
}).passthrough();

const SourceModel = z.object({
  catalog: CatalogModel,
  schemaInput: z.unknown().optional(),
});

const CuratedModel = z.object({
  base_model: z.string().min(1).optional(),
  structured_output: z.boolean().optional(),
  reasoning_options: z.array(ReasoningOption).optional(),
  limit: z.object({
    context: z.number().optional(),
    input: z.number().optional(),
    output: z.number().optional(),
  }).strict().optional(),
  interleaved: z.union([
    z.literal(true),
    z.object({ field: z.enum(["reasoning_content", "reasoning_details"]) }).strict(),
  ]).optional(),
  note: z.array(z.string()).optional(),
}).strict();

const Curation = z.object({
  skip: z.array(z.string()).default([]),
  models: z.record(CuratedModel).default({}),
}).strict();

type CatalogModel = z.infer<typeof CatalogModel>;
type CatalogEntry = z.infer<typeof CatalogEntry>;
type SourceModel = z.infer<typeof SourceModel>;
type CuratedModel = z.infer<typeof CuratedModel>;

const curation = Curation.parse(Bun.TOML.parse(readFileSync(CURATION_PATH, "utf8")));
const skippedModels = new Set(curation.skip);

export const cloudflareAiGateway = {
  id: "cloudflare-ai-gateway",
  name: "Cloudflare AI Gateway",
  modelsDir: "providers/cloudflare-ai-gateway/models",
  preserveDescriptions: false,
  authoritativeHeaders: true,
  async fetchModels() {
    const catalog = CatalogEntry.array().parse(await loadCatalog());
    const catalogIDs = new Set(catalog.map((model) => model.model_id));
    if (catalogIDs.size !== catalog.length) {
      throw new Error("Cloudflare AI Gateway catalog returned duplicate model IDs");
    }
    const textModels = catalog
      .filter((model) => model.task === TEXT_GENERATION)
      .map((model) => CatalogModel.parse(model));
    if (textModels.length === 0) {
      throw new Error("Cloudflare AI Gateway catalog returned no Text Generation models");
    }

    const emittedModels = textModels.filter((model) => !skippedModels.has(model.model_id));
    const sources = await mapLimit(emittedModels, 6, async (model) => ({
      catalog: model,
      schemaInput: await loadCatalogSchemaInput(model.model_id),
    }));

    const liveIDs = new Set(textModels.map((model) => model.model_id));
    for (const id of Object.keys(curation.models)) {
      if (!liveIDs.has(id)) console.warn(`warning: curation id not in live feed: ${id}`);
    }

    return sources;
  },
  parseModels(raw) {
    return SourceModel.array().parse(raw);
  },
  translateModel(source, context) {
    const id = source.catalog.model_id;
    const curated = curation.models[id] ?? {};
    return {
      id,
      model: buildCloudflareAiGatewayModel(
        source.catalog,
        source.schemaInput,
        curated,
        context.authored(id),
      ),
      header: noteHeader(curated.note),
    };
  },
} satisfies SyncProvider<SourceModel>;

export function buildCloudflareAiGatewayModel(
  catalog: CatalogModel,
  schemaInput: unknown,
  curated: CuratedModel = {},
  existing?: ExistingModel,
): SyncedBaseModel {
  const id = catalog.model_id;
  const baseModel = curated.base_model ?? resolveBaseModel(id);
  if (baseModel === undefined) {
    throw new Error(`${id}: no lab file and no curated base_model; add it to skip or map it`);
  }

  const model: SyncedBaseModel = { base_model: baseModel };
  if (curated.structured_output !== undefined) {
    model.structured_output = curated.structured_output;
  }
  if (curated.interleaved !== undefined) model.interleaved = curated.interleaved;

  if (baseReasoning(baseModel)) {
    const derived = deriveReasoningOptions(schemaInput);
    const reasoningOptions = curated.reasoning_options ?? (derived.length > 0 ? derived : undefined);
    if (reasoningOptions === undefined) {
      throw new Error(
        `${id}: base ${baseModel} reasons but the catalog schema and curation provide no reasoning_options`,
      );
    }
    model.reasoning_options = reasoningOptions;
  }

  model.cost = proxiedCost(catalog.pricing, id);

  const limit = {
    ...(catalog.context_length == null && existing?.limit?.context === undefined
      ? {}
      : { context: catalog.context_length ?? existing?.limit?.context }),
    ...curated.limit,
  };
  if (Object.keys(limit).length > 0) model.limit = limit;

  const npm = NATIVE_NPM[id.split("/")[0]!];
  if (npm !== undefined) model.provider = { npm };
  return model;
}

export function deriveReasoningOptions(
  schemaInput: unknown,
): NonNullable<SyncedBaseModel["reasoning_options"]> {
  let hasToggle = false;
  let effortValues: Array<"none" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max" | "default">
    | undefined;

  const EffortValues = z.array(z.enum([
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
    "default",
  ]));

  const visit = (node: unknown) => {
    if (Array.isArray(node)) {
      node.forEach(visit);
      return;
    }
    if (node === null || typeof node !== "object") return;

    for (const [key, value] of Object.entries(node)) {
      if (key !== "properties" || value === null || typeof value !== "object") {
        visit(value);
        continue;
      }

      for (const [property, rawSchema] of Object.entries(value)) {
        const propertySchema = rawSchema as Record<string, unknown>;
        if (property === "enable_thinking" || property === "thinking") hasToggle = true;
        if (property === "effort" || property === "reasoning_effort") {
          const candidates = [propertySchema, ...arrayValue(propertySchema.anyOf), ...arrayValue(propertySchema.oneOf)];
          for (const candidate of candidates) {
            const parsed = EffortValues.safeParse(candidate.enum);
            if (parsed.success) effortValues = parsed.data;
          }
        }
        visit(rawSchema);
      }
    }
  };
  visit(schemaInput);

  const options: NonNullable<SyncedBaseModel["reasoning_options"]> = [];
  if (hasToggle) options.push({ type: "toggle" });
  if (effortValues !== undefined) options.push({ type: "effort", values: effortValues });
  return options;
}

function arrayValue(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => item !== null && typeof item === "object")
    : [];
}

async function loadCatalog() {
  const fixtureDir = process.env.CF_AIG_FIXTURE_DIR;
  if (fixtureDir !== undefined) return loadFixtureRows(fixtureDir, "catalog");

  const { accountID, token } = credentials();
  const models: CatalogEntry[] = [];
  let expectedTotal: number | undefined;
  for (let page = 1; page <= 1_000; page++) {
    const url = new URL(`${API_BASE}/${accountID}/ai/catalog/models`);
    url.searchParams.set("page", String(page));
    url.searchParams.set("per_page", "50");
    const response = await fetchWithRetry(url, { headers: { Authorization: `Bearer ${token}` } });
    if (!response.ok) {
      throw new Error(`Cloudflare AI Gateway catalog request failed: ${response.status} ${response.statusText}`);
    }
    const body = CloudflareResponse.parse(await response.json());
    expectedTotal ??= body.result_info.total_count;
    if (body.result_info.total_count !== expectedTotal) {
      throw new Error("Cloudflare AI Gateway catalog total changed during pagination");
    }
    models.push(...body.result);
    if (models.length === expectedTotal) return models;
    if (body.result.length === 0 || models.length > expectedTotal) {
      throw new Error(`Cloudflare AI Gateway catalog pagination ended at ${models.length}/${expectedTotal}`);
    }
  }
  throw new Error("Cloudflare AI Gateway catalog exceeded the pagination safety limit");
}

async function loadCatalogSchemaInput(id: string): Promise<unknown> {
  const fixtureDir = process.env.CF_AIG_FIXTURE_DIR;
  if (fixtureDir !== undefined) {
    const file = path.join(fixtureDir, "schema", `${id.replaceAll("/", "_")}.json`);
    if (!existsSync(file)) return undefined;
    return z.object({
      result: z.object({ schema: z.object({ input: z.unknown().optional() }).passthrough() }).passthrough(),
    }).passthrough().parse(JSON.parse(readFileSync(file, "utf8"))).result.schema.input;
  }

  const { accountID, token } = credentials();
  const response = await fetchWithRetry(
    `${API_BASE}/${accountID}/ai/catalog/models/${id.split("/").map(encodeURIComponent).join("/")}/schema`,
    { headers: { Authorization: `Bearer ${token}` } },
  );
  if (!response.ok) return undefined;
  return z.object({
    result: z.object({ schema: z.object({ input: z.unknown().optional() }).passthrough() }).passthrough(),
  }).passthrough().parse(await response.json()).result.schema.input;
}

function credentials() {
  const token = process.env.CLOUDFLARE_API_TOKEN
    ?? process.env.CLOUDFLARE_PRODUCTION_API_TOKEN;
  const accountID = process.env.CLOUDFLARE_ACCOUNT_ID
    ?? process.env.CLOUDFLARE_PRODUCTION_ACCOUNT_ID_AI_GATEWAY_SANDBOX;
  if (!token || !accountID) {
    throw new Error(
      "Cloudflare AI Gateway sync requires Cloudflare API token and account ID credentials",
    );
  }
  return { accountID, token };
}

function loadFixtureRows(directory: string, prefix: string): unknown[] {
  const rows = new Map<string, unknown>();
  let expectedTotal: number | undefined;
  for (const file of readdirSync(directory).filter((name) => name.startsWith(prefix) && name.endsWith(".json"))) {
    const response = CloudflareResponse.parse(JSON.parse(readFileSync(path.join(directory, file), "utf8")));
    expectedTotal ??= response.result_info.total_count;
    if (response.result_info.total_count !== expectedTotal) {
      throw new Error("Cloudflare AI Gateway fixtures have inconsistent catalog totals");
    }
    for (const model of response.result) rows.set(model.model_id, model);
  }
  if (rows.size !== expectedTotal) {
    throw new Error(`Cloudflare AI Gateway fixtures contain ${rows.size}/${expectedTotal ?? 0} catalog models`);
  }
  return [...rows.values()];
}

async function fetchWithRetry(url: string | URL, init: RequestInit, attempts = 5): Promise<Response> {
  let delay = 500;
  for (let attempt = 1; ; attempt++) {
    const response = await fetch(url, { ...init, signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS) });
    if (response.ok || (response.status !== 429 && response.status < 500) || attempt >= attempts) {
      return response;
    }
    const retryAfter = Number(response.headers.get("retry-after"));
    const wait = Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter * 1000 : delay;
    await Bun.sleep(Math.min(wait, 8_000));
    delay = Math.min(delay * 2, 8_000);
  }
}

async function mapLimit<T, R>(items: T[], limit: number, transform: (item: T) => Promise<R>) {
  const results = new Array<R>(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      results[index] = await transform(items[index]!);
    }
  }));
  return results;
}

const FLAT_PRICING_KEYS: Record<string, "input" | "output" | "cache_read" | "cache_write"> = {
  "Input tokens (per 1M)": "input",
  "Output tokens (per 1M)": "output",
  "Cached input tokens (per 1M)": "cache_read",
  "Cache creation tokens (per 1M)": "cache_write",
};
const TIERED_PRICING_KEY = /^(Input|Output|Cached input)\s*(<=?|>=?)\s*(\d+)k\s*\(per 1M\)$/;
const TIERED_PRICING_FIELDS = {
  Input: "input",
  Output: "output",
  "Cached input": "cache_read",
} as const;

function proxiedCost(pricing: Record<string, number>, id: string): NonNullable<SyncedBaseModel["cost"]> {
  const cost: NonNullable<SyncedBaseModel["cost"]> = {};
  for (const [key, value] of Object.entries(pricing)) {
    const flatField = FLAT_PRICING_KEYS[key];
    if (flatField !== undefined) {
      cost[flatField] = value;
      continue;
    }
    const tier = TIERED_PRICING_KEY.exec(key);
    if (tier !== null) {
      const field = TIERED_PRICING_FIELDS[tier[1] as keyof typeof TIERED_PRICING_FIELDS];
      if (tier[2]!.startsWith("<")) cost[field] = value;
      continue;
    }
    throw new Error(`${id}: unmapped pricing key "${key}"`);
  }
  if (cost.input === undefined || cost.output === undefined) {
    throw new Error(`${id}: catalog pricing must include input and output rates`);
  }
  return cost;
}

function isSafeModelID(id: string) {
  if (path.isAbsolute(id) || id.includes("\\")) return false;
  const segments = id.split("/");
  return segments.length >= 2
    && segments.every((segment) => /^[A-Za-z0-9@._-]+$/.test(segment) && segment !== "." && segment !== "..");
}

function resolveBaseModel(id: string) {
  if (labFileExists(id)) return id;
  const dashed = id.replaceAll(".", "-");
  return labFileExists(dashed) ? dashed : undefined;
}

function labFileExists(id: string) {
  return existsSync(path.join(MODELS_ROOT, `${id}.toml`));
}

function baseReasoning(id: string) {
  const file = path.join(MODELS_ROOT, `${id}.toml`);
  return existsSync(file) && z.object({ reasoning: z.boolean().optional() }).passthrough()
    .parse(Bun.TOML.parse(readFileSync(file, "utf8"))).reasoning === true;
}

function noteHeader(note: string[] | undefined) {
  return note === undefined || note.length === 0
    ? undefined
    : `${note.map((line) => `# ${line}`).join("\n")}\n`;
}
