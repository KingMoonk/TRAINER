"use client";

// A script that must run before first paint. It runs from the prerendered HTML; React never runs a <script> it creates on
// the client, and React 19 warns when it creates one ("Encountered a script tag..."). So on the client the same element
// gets type="text/plain", which React treats as a data block: no warning, and nothing changes on the page.
export const CLIENT_SCRIPT_TYPE = typeof window === "undefined" ? undefined : "text/plain";

export function InlineScript({ code }: { code: string }) {
  return <script type={CLIENT_SCRIPT_TYPE} suppressHydrationWarning dangerouslySetInnerHTML={{ __html: code }} />;
}
