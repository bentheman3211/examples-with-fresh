// routes/nsfw.ts
import type { Handlers } from "$fresh/server.ts";
import * as nsfwjs from "https://esm.sh/nsfwjs@2.4.0";
import * as tf from "https://esm.sh/@tensorflow/tfjs-node@4.12.0";
import { decode } from "https://deno.land/x/fast_image@0.1.0/mod.ts";

let model: nsfwjs.NSFWJS | null = null;

async function loadModel() {
  if (!model) {
    console.log("Loading NSFWJS model...");
    model = await nsfwjs.load();
    console.log("Model loaded!");
  }
}

await loadModel();

export const handler: Handlers = {
  async POST(req) {
    try {
      const body = await req.json();
      const imageUrl = body.url;
      if (!imageUrl) return new Response(JSON.stringify({ error: "Missing url" }), { status: 400 });

      const resp = await fetch(imageUrl);
      if (!resp.ok) throw new Error("Failed to fetch image");
      const buffer = new Uint8Array(await resp.arrayBuffer());

      const imgTensor = await decode(buffer);
      const predictions = await model!.classify(imgTensor);
      imgTensor.dispose();

      return new Response(JSON.stringify(predictions), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (err) {
      console.error(err);
      return new Response(JSON.stringify({ error: err.message }), { status: 500 });
    }
  }
};
