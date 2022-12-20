import { serve } from "https://deno.land/std@0.160.0/http/server.ts";
import { serveFile } from "https://deno.land/std@0.160.0/http/file_server.ts";
import { existsSync } from "https://deno.land/std@0.160.0/fs/exists.ts";

const handleRequest = async (request) => {
  const url = new URL(request.url);
  const path = `${Deno.cwd()}/static${url.pathname}`;
  if (!request.url.endsWith("/") && existsSync(path)) {
    return await serveFile(request, path);
  } else {
    return new Response("Not found", { status: 404 });
  }
};

serve(handleRequest, { port: 7777 });