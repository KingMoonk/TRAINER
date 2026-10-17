// The Funnel token the API wants in `X-Groundwork-Token` (docs/node.md). It lives in this browser's localStorage, set once by
// the user, never in the build (a NEXT_PUBLIC_ value is readable by anyone who loads the site) and never in a URL.

export const TOKEN_KEY = "groundwork-token";
export const TOKEN_HEADER = "X-Groundwork-Token";

/** The stored token, or "" when there is none or storage is unavailable. */
export function apiToken(): string {
  try {
    return globalThis.localStorage?.getItem(TOKEN_KEY)?.trim() ?? "";
  } catch {
    return "";
  }
}

/** `{ "X-Groundwork-Token": token }`, or `{}` when no token is stored. Spread it into a request's headers. */
export function authHeaders(): Record<string, string> {
  const token = apiToken();
  return token ? { [TOKEN_HEADER]: token } : {};
}
