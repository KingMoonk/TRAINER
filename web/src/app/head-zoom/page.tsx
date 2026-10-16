import { Suspense } from "react";
import { HeadZoom } from "@/app/head-zoom/head-zoom";

// HeadZoom reads ?block=&head= (a link from the Head lab opens one head), so it sits in a Suspense boundary for the static export.
export default function Page() {
  return (
    <Suspense>
      <HeadZoom />
    </Suspense>
  );
}
