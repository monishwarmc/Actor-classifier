import { Client, handle_file } from "@gradio/client";
import formidable from "formidable";
import fs from "fs/promises";

const HF_SPACE = "monishwar369/Actor_classifier";
const HF_ENDPOINT = "/predict_actor";

export const config = {
  api: {
    bodyParser: false,
  },
};

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({
      error: "Method not allowed",
    });
  }

  try {
    if (!process.env.HF_TOKEN) {
      console.error("HF_TOKEN is missing");

      return res.status(500).json({
        error: "HF_TOKEN is not configured on Vercel",
      });
    }

    const form = formidable({
      multiples: false,
      maxFileSize: 10 * 1024 * 1024,
    });

    const [, files] = await form.parse(req);

    const uploadedFile = Array.isArray(files.image)
      ? files.image[0]
      : files.image;

    if (!uploadedFile) {
      return res.status(400).json({
        error: "No image uploaded",
      });
    }

    console.log("Received image:", uploadedFile.originalFilename);
    console.log("Connecting to Hugging Face...");

    const client = await Client.connect(HF_SPACE, {
      token: process.env.HF_TOKEN,
    });

    const imageBuffer = await fs.readFile(uploadedFile.filepath);

    const imageBlob = new Blob([imageBuffer], {
      type: uploadedFile.mimetype || "image/jpeg",
    });

    console.log("Sending image to Hugging Face...");

    const result = await client.predict(HF_ENDPOINT, {
      input_image: handle_file(imageBlob),
    });

    console.log("Hugging Face response:", result.data);

    return res.status(200).json(result.data);
  } catch (error) {
    console.error("Prediction error:", error);

    return res.status(500).json({
      error: error?.message || "Prediction failed",
    });
  }
}
