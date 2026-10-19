"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState } from "react";
import { AppearanceProvider } from "@/components/theme";

export function Providers({ children }: { children: React.ReactNode }) {
  const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { staleTime: 5_000 } } }));
  return (
    <AppearanceProvider>
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    </AppearanceProvider>
  );
}
