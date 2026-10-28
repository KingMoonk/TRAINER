// The "Developer tools" setting: whether the sidebar shows the API traffic console button. Off by default and remembered per browser
// like the reply voice. Ctrl+` opens the console either way (components/traffic/traffic-provider.tsx).

import { useCallback, useSyncExternalStore } from "react";

export const DEVELOPER_TOOLS_KEY = "settings.developerTools";

const watchers = new Set<() => void>();
function watch(notify: () => void) {
  watchers.add(notify);
  window.addEventListener("storage", notify);
  return () => {
    watchers.delete(notify);
    window.removeEventListener("storage", notify);
  };
}

function read(): boolean {
  try {
    return localStorage.getItem(DEVELOPER_TOOLS_KEY) === "on";
  } catch {
    return false; // storage can be blocked; the setting then stays off
  }
}

/** Whether developer tools are on (false on the server and in the first render), and a setter that remembers the choice. */
export function useDeveloperTools(): [boolean, (on: boolean) => void] {
  const on = useSyncExternalStore(watch, read, () => false);
  const set = useCallback((next: boolean) => {
    try {
      localStorage.setItem(DEVELOPER_TOOLS_KEY, next ? "on" : "off");
    } catch {
      // Not remembered; the setting stays as it was.
    }
    watchers.forEach((notify) => notify());
  }, []);
  return [on, set];
}
