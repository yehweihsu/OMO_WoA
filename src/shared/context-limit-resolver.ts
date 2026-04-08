import process from "node:process"

const DEFAULT_ANTHROPIC_ACTUAL_LIMIT = 200_000
export type ContextLimitModelCacheState = {
  anthropicContext1MEnabled: boolean
  modelContextLimitsCache?: Map<string, number>
}

function isAnthropicProvider(providerID: string): boolean {
  const normalized = providerID.toLowerCase()
  return normalized === "anthropic" || normalized === "google-vertex-anthropic" || normalized === "aws-bedrock-anthropic"
}

function getAnthropicActualLimit(modelCacheState?: ContextLimitModelCacheState): number {
  return (modelCacheState?.anthropicContext1MEnabled ?? false) ||
    process.env.ANTHROPIC_1M_CONTEXT === "true" ||
    process.env.VERTEX_ANTHROPIC_1M_CONTEXT === "true"
    ? 1_000_000
    : DEFAULT_ANTHROPIC_ACTUAL_LIMIT
}

function supportsCachedAnthropicLimit(modelID: string): boolean {
  return /^claude-(opus|sonnet)-4(?:-|\.)6(?:-high)?$/.test(modelID)
}

/**
 * Known context window limits for providers that don't report them via API.
 * These are used as fallbacks when the model cache doesn't have the limit.
 * Values are in tokens.
 */
const KNOWN_PROVIDER_LIMITS: Record<string, Record<string, number>> = {
  // OpenCode Go provider (opencode-go/)
  "opencode-go": {
    "glm-5.1": 128_000,
    "glm-5": 128_000,
    "kimi-k2.5": 128_000,
    "minimax-m2.7": 32_000,
    "minimax-m2.5": 32_000,
    "mimo-v2-pro": 64_000,
    "mimo-v2-omni": 64_000,
  },
  // OpenCode Zen provider (opencode/)
  "opencode": {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "gpt-5.4": 128_000,
    "gpt-5": 128_000,
    "gemini-3.1-pro": 1_000_000,
    "gemini-3-flash": 1_000_000,
    "minimax-m2.5": 32_000,
    "minimax-m2.5-free": 32_000,
    "glm-5.1": 128_000,
    "kimi-k2.5": 128_000,
    "big-pickle": 128_000,
  },
  // Other common providers
  "openai": {
    "gpt-5.4": 128_000,
    "gpt-5": 128_000,
  },
  "google": {
    "gemini-3.1-pro": 1_000_000,
    "gemini-3-flash": 1_000_000,
  },
}

/**
 * Resolve context window limit for a model.
 * Priority: API cache > Anthropic explicit > Known limits > null
 */
export function resolveActualContextLimit(
  providerID: string,
  modelID: string,
  modelCacheState?: ContextLimitModelCacheState,
): number | null {
  if (isAnthropicProvider(providerID)) {
    const explicit1M = getAnthropicActualLimit(modelCacheState)
    if (explicit1M === 1_000_000) return explicit1M

    const cachedLimit = modelCacheState?.modelContextLimitsCache?.get(`${providerID}/${modelID}`)
    if (cachedLimit && supportsCachedAnthropicLimit(modelID)) return cachedLimit

    return DEFAULT_ANTHROPIC_ACTUAL_LIMIT
  }

  // Check runtime cache first
  const cachedLimit = modelCacheState?.modelContextLimitsCache?.get(`${providerID}/${modelID}`)
  if (cachedLimit) return cachedLimit

  // Fall back to known limits
  const providerLimits = KNOWN_PROVIDER_LIMITS[providerID]
  if (providerLimits) {
    // Try exact match first
    if (providerLimits[modelID] !== undefined) return providerLimits[modelID]

    // Try prefix match (handles variant IDs like "glm-5.1-0614")
    for (const [knownModel, limit] of Object.entries(providerLimits)) {
      if (modelID.startsWith(knownModel)) return limit
    }
  }

  return null
}

/**
 * Get the appropriate tool output truncation limit based on model context window.
 * Models with smaller context windows get more aggressive truncation.
 */
export function getToolOutputMaxTokens(contextLimit: number | null): number {
  if (contextLimit === null) return 50_000 // Unknown model, conservative default

  if (contextLimit <= 40_000) return 8_000   // Small context (minimax-m2.7 ~32K)
  if (contextLimit <= 80_000) return 15_000   // Medium context (mimo ~64K)
  if (contextLimit <= 160_000) return 20_000  // Large context (glm-5, gpt-5 ~128K)
  return 50_000                                // Very large context (claude, gemini ~200K+)
}
