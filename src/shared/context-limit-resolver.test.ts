import process from "node:process"
import { afterEach, describe, expect, it } from "bun:test"

import { resolveActualContextLimit, getToolOutputMaxTokens } from "./context-limit-resolver"

const ANTHROPIC_CONTEXT_ENV_KEY = "ANTHROPIC_1M_CONTEXT"
const VERTEX_CONTEXT_ENV_KEY = "VERTEX_ANTHROPIC_1M_CONTEXT"

const originalAnthropicContextEnv = process.env[ANTHROPIC_CONTEXT_ENV_KEY]
const originalVertexContextEnv = process.env[VERTEX_CONTEXT_ENV_KEY]

function resetContextLimitEnv(): void {
  if (originalAnthropicContextEnv === undefined) {
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
  } else {
    process.env[ANTHROPIC_CONTEXT_ENV_KEY] = originalAnthropicContextEnv
  }

  if (originalVertexContextEnv === undefined) {
    delete process.env[VERTEX_CONTEXT_ENV_KEY]
  } else {
    process.env[VERTEX_CONTEXT_ENV_KEY] = originalVertexContextEnv
  }
}

describe("resolveActualContextLimit", () => {
  afterEach(() => {
    resetContextLimitEnv()
  })

  it("returns cached limit for Anthropic 4.6 models when 1M mode is disabled (GA support)", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]
    const modelContextLimitsCache = new Map<string, number>()
    modelContextLimitsCache.set("anthropic/claude-opus-4-6", 1_000_000)

    // when
    const actualLimit = resolveActualContextLimit("anthropic", "claude-opus-4-6", {
      anthropicContext1MEnabled: false,
      modelContextLimitsCache,
    })

    expect(actualLimit).toBe(1_000_000)
  })

  it("returns default 200K for older Anthropic models when 1M mode is disabled", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]
    const modelContextLimitsCache = new Map<string, number>()
    modelContextLimitsCache.set("anthropic/claude-sonnet-4-5", 500_000)

    // when
    const actualLimit = resolveActualContextLimit("anthropic", "claude-sonnet-4-5", {
      anthropicContext1MEnabled: false,
      modelContextLimitsCache,
    })

    // then
    expect(actualLimit).toBe(200_000)
  })

  it("returns default 200K for Anthropic models without cached limit and 1M mode disabled", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]

    // when
    const actualLimit = resolveActualContextLimit("anthropic", "claude-sonnet-4-5", {
      anthropicContext1MEnabled: false,
    })

    // then
    expect(actualLimit).toBe(200_000)
  })

  it("explicit 1M mode takes priority over cached limit", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]
    const modelContextLimitsCache = new Map<string, number>()
    modelContextLimitsCache.set("anthropic/claude-sonnet-4-5", 200_000)

    // when
    const actualLimit = resolveActualContextLimit("anthropic", "claude-sonnet-4-5", {
      anthropicContext1MEnabled: true,
      modelContextLimitsCache,
    })

    expect(actualLimit).toBe(1_000_000)
  })

  it("treats Anthropics aliases as Anthropic providers", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]

    // when
    const actualLimit = resolveActualContextLimit(
      "aws-bedrock-anthropic",
      "claude-sonnet-4-5",
      { anthropicContext1MEnabled: false },
    )

    // then
    expect(actualLimit).toBe(200000)
  })

  it("supports Anthropic 4.6 dot-version model IDs without explicit 1M mode", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]
    const modelContextLimitsCache = new Map<string, number>()
    modelContextLimitsCache.set("anthropic/claude-opus-4.6", 1_000_000)

    // when
    const actualLimit = resolveActualContextLimit("anthropic", "claude-opus-4.6", {
      anthropicContext1MEnabled: false,
      modelContextLimitsCache,
    })

    // then
    expect(actualLimit).toBe(1_000_000)
  })

  it("supports Anthropic 4.6 high-variant model IDs without widening older models", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]
    const modelContextLimitsCache = new Map<string, number>()
    modelContextLimitsCache.set("anthropic/claude-sonnet-4-6-high", 500_000)

    // when
    const actualLimit = resolveActualContextLimit("anthropic", "claude-sonnet-4-6-high", {
      anthropicContext1MEnabled: false,
      modelContextLimitsCache,
    })

    // then
    expect(actualLimit).toBe(500_000)
  })

  it("ignores stale cached limits for older Anthropic models with suffixed IDs", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]
    const modelContextLimitsCache = new Map<string, number>()
    modelContextLimitsCache.set("anthropic/claude-sonnet-4-5-high", 500_000)

    // when
    const actualLimit = resolveActualContextLimit("anthropic", "claude-sonnet-4-5-high", {
      anthropicContext1MEnabled: false,
      modelContextLimitsCache,
    })

    // then
    expect(actualLimit).toBe(200_000)
  })

  it("returns null for non-Anthropic providers without a cached limit and without known limits", () => {
    // given
    delete process.env[ANTHROPIC_CONTEXT_ENV_KEY]
    delete process.env[VERTEX_CONTEXT_ENV_KEY]

    // when
    const actualLimit = resolveActualContextLimit("some-unknown-provider", "some-model", {
      anthropicContext1MEnabled: false,
    })

    // then
    expect(actualLimit).toBeNull()
  })

  // #given OpenCode Go provider with known models
  it("returns known context limit for OpenCode Go models", () => {
    // when
    const glm5Limit = resolveActualContextLimit("opencode-go", "glm-5.1", {
      anthropicContext1MEnabled: false,
    })
    const minimaxLimit = resolveActualContextLimit("opencode-go", "minimax-m2.7", {
      anthropicContext1MEnabled: false,
    })
    const kimiLimit = resolveActualContextLimit("opencode-go", "kimi-k2.5", {
      anthropicContext1MEnabled: false,
    })

    // then
    expect(glm5Limit).toBe(128_000)
    expect(minimaxLimit).toBe(32_000)
    expect(kimiLimit).toBe(128_000)
  })

  // #given OpenCode Go model with prefix match
  it("matches model variants via prefix matching", () => {
    // when
    const limit = resolveActualContextLimit("opencode-go", "glm-5.1-0614", {
      anthropicContext1MEnabled: false,
    })

    // then — variant ID should match the base model prefix
    expect(limit).toBe(128_000)
  })

  // #given OpenCode Zen provider with known models
  it("returns known context limit for OpenCode Zen models", () => {
    // when
    const claudeLimit = resolveActualContextLimit("opencode", "claude-opus-4-6", {
      anthropicContext1MEnabled: false,
    })
    const geminiLimit = resolveActualContextLimit("opencode", "gemini-3.1-pro", {
      anthropicContext1MEnabled: false,
    })

    // then
    expect(claudeLimit).toBe(200_000)
    expect(geminiLimit).toBe(1_000_000)
  })

  // #given cached limit takes priority over known limits
  it("cached limit takes priority over known limits for non-Anthropic providers", () => {
    // given
    const modelContextLimitsCache = new Map<string, number>()
    modelContextLimitsCache.set("opencode-go/glm-5.1", 256_000)

    // when
    const actualLimit = resolveActualContextLimit("opencode-go", "glm-5.1", {
      anthropicContext1MEnabled: false,
      modelContextLimitsCache,
    })

    // then — runtime cache overrides known limits
    expect(actualLimit).toBe(256_000)
  })
})

describe("getToolOutputMaxTokens", () => {
  // #given null context limit (unknown model)
  it("returns 50000 for unknown context limit", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(null)

    // then — conservative default for unknown models
    expect(maxTokens).toBe(50_000)
  })

  // #given small context models (minimax-m2.7 ~32K)
  it("returns 8000 for small context models under 40K", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(32_000)

    // then — aggressive truncation for small context
    expect(maxTokens).toBe(8_000)
  })

  // #given medium context models (mimo ~64K)
  it("returns 15000 for medium context models", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(64_000)

    // then — moderate truncation
    expect(maxTokens).toBe(15_000)
  })

  // #given large context models (glm-5.1, gpt-5 ~128K)
  it("returns 20000 for large context models", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(128_000)

    // then — moderate truncation for 128K context
    expect(maxTokens).toBe(20_000)
  })

  // #given very large context models (claude, gemini ~200K+)
  it("returns 50000 for very large context models", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(200_000)

    // then — less aggressive truncation for high-limit models
    expect(maxTokens).toBe(50_000)
  })

  // #given boundary value at 40K
  it("returns 8000 at exact 40K boundary (falls into small context tier)", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(40_000)

    // then — 40K equals the boundary, falls into <= 40K tier
    expect(maxTokens).toBe(8_000)
  })

  // #given boundary value at 80K
  it("returns 15000 at exact 80K boundary (falls into medium context tier)", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(80_000)

    // then
    expect(maxTokens).toBe(15_000)
  })

  // #given boundary value at 160K
  it("returns 20000 at exact 160K boundary (falls into large context tier)", () => {
    // when
    const maxTokens = getToolOutputMaxTokens(160_000)

    // then
    expect(maxTokens).toBe(20_000)
  })
})
