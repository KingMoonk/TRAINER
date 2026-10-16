import { PagePlaceholder } from "@/components/page-placeholder";
import { routeFor } from "@/lib/routes";

export default function Page() {
  return <PagePlaceholder route={routeFor("make_data")} />;
}
