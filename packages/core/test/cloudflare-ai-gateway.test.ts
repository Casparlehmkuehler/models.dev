import { expect, test } from "bun:test";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";

import { syncProvider } from "../src/sync/index.js";
import {
  buildCloudflareAiGatewayModel,
  cloudflareAiGateway,
  deriveReasoningOptions,
} from "../src/sync/providers/cloudflare-ai-gateway.js";

test("builds Cloudflare AI Gateway overrides from catalog metadata", () => {
  const model = buildCloudflareAiGatewayModel(
    {
      model_id: "openai/gpt-5.4",
      task: "Text Generation",
      context_length: 1_050_000,
      pricing: {
        "Input <= 200k (per 1M)": 2.5,
        "Input > 200k (per 1M)": 5,
        "Output tokens (per 1M)": 15,
        "Cached input tokens (per 1M)": 0.25,
      },
    },
    undefined,
    {
      reasoning_options: [{ type: "effort", values: ["none", "low", "medium", "high", "xhigh"] }],
    },
  );

  expect(model).toEqual({
    base_model: "openai/gpt-5.4",
    reasoning_options: [{ type: "effort", values: ["none", "low", "medium", "high", "xhigh"] }],
    cost: { input: 2.5, output: 15, cache_read: 0.25 },
    limit: { context: 1_050_000 },
    provider: { npm: "@ai-sdk/openai" },
  });
});

test("derives nested Cloudflare reasoning controls", () => {
  expect(deriveReasoningOptions({
    properties: {
      thinking: { type: "boolean" },
      reasoning: {
        properties: {
          effort: {
            anyOf: [{ enum: ["low", "medium", "high"] }],
          },
        },
      },
    },
  })).toEqual([
    { type: "toggle" },
    { type: "effort", values: ["low", "medium", "high"] },
  ]);
});

test("ignores advertised reasoning controls for non-reasoning base models", () => {
  const model = buildCloudflareAiGatewayModel(
    {
      model_id: "openai/gpt-4.1",
      task: "Text Generation",
      context_length: 1_047_576,
      pricing: {
        "Input tokens (per 1M)": 2,
        "Output tokens (per 1M)": 8,
      },
    },
    {
      properties: {
        reasoning_effort: { enum: ["low", "medium", "high"] },
      },
    },
  );

  expect(model.reasoning_options).toBeUndefined();
});

test("fails closed on unknown pricing fields", () => {
  expect(() => buildCloudflareAiGatewayModel({
    model_id: "openai/gpt-4.1",
    task: "Text Generation",
    context_length: 1_047_576,
    pricing: {
      "Input tokens (per 1M)": 2,
      "Output tokens (per 1M)": 8,
      "New billing unit": 1,
    },
  }, undefined)).toThrow('unmapped pricing key "New billing unit"');
});

test("fails closed when Cloudflare pagination is incomplete", async () => {
  const originalFetch = globalThis.fetch;
  const originalToken = process.env.CLOUDFLARE_API_TOKEN;
  const originalAccount = process.env.CLOUDFLARE_ACCOUNT_ID;
  process.env.CLOUDFLARE_API_TOKEN = "test";
  process.env.CLOUDFLARE_ACCOUNT_ID = "test";
  let page = 0;
  globalThis.fetch = async () => new Response(JSON.stringify({
    result: page++ === 0
      ? [{
        model_id: "openai/gpt-4.1",
        task: "Text Generation",
        context_length: 1_047_576,
        pricing: {
          "Input tokens (per 1M)": 2,
          "Output tokens (per 1M)": 8,
        },
      }]
      : [],
    result_info: { total_count: 2 },
  }));

  try {
    await expect(cloudflareAiGateway.fetchModels()).rejects.toThrow("pagination ended at 1/2");
  } finally {
    globalThis.fetch = originalFetch;
    restoreEnv("CLOUDFLARE_API_TOKEN", originalToken);
    restoreEnv("CLOUDFLARE_ACCOUNT_ID", originalAccount);
  }
});

test("rejects unsafe catalog model paths", async () => {
  const originalFetch = globalThis.fetch;
  const originalToken = process.env.CLOUDFLARE_API_TOKEN;
  const originalAccount = process.env.CLOUDFLARE_ACCOUNT_ID;
  process.env.CLOUDFLARE_API_TOKEN = "test";
  process.env.CLOUDFLARE_ACCOUNT_ID = "test";
  globalThis.fetch = async () => new Response(JSON.stringify({
    result: [{
      model_id: "../providers/openai/models/gpt-4.1",
      task: "Text Generation",
      context_length: 1_047_576,
      pricing: {
        "Input tokens (per 1M)": 2,
        "Output tokens (per 1M)": 8,
      },
    }],
    result_info: { total_count: 1 },
  }));

  try {
    await expect(cloudflareAiGateway.fetchModels()).rejects.toThrow("safe relative provider/model path");
  } finally {
    globalThis.fetch = originalFetch;
    restoreEnv("CLOUDFLARE_API_TOKEN", originalToken);
    restoreEnv("CLOUDFLARE_ACCOUNT_ID", originalAccount);
  }
});

test("reconciles authoritative generated headers", async () => {
  const providersDir = path.join(import.meta.dirname, "..", "..", "..", "providers");
  const providerDir = await mkdtemp(path.join(providersDir, ".cloudflare-ai-gateway-sync-"));
  const modelsDir = path.join(providerDir, "models");
  await mkdir(modelsDir);
  const file = path.join(modelsDir, "gpt-4.1.toml");
  await writeFile(file, "# Old note\n\nbase_model = \"openai/gpt-4.1\"\n");

  try {
    const provider = {
      id: "cloudflare-ai-gateway-test",
      name: "Cloudflare AI Gateway test",
      modelsDir,
      authoritativeHeaders: true,
      async fetchModels() {
        return [{ id: "gpt-4.1" }];
      },
      parseModels() {
        return [{ id: "gpt-4.1" }];
      },
      translateModel(model) {
        return {
          id: model.id,
          model: { base_model: "openai/gpt-4.1" },
          header: "# New note\n\n",
        };
      },
    };
    const result = await syncProvider(provider);

    expect(result.updated).toBe(1);
    expect(await readFile(file, "utf8")).toStartWith("# New note\nbase_model");
    expect((await syncProvider(provider)).updated).toBe(0);
  } finally {
    await rm(providerDir, { recursive: true, force: true });
  }
});

function restoreEnv(name: string, value: string | undefined) {
  if (value === undefined) delete process.env[name];
  else process.env[name] = value;
}
