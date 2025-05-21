import requests
import base64
from typing import Any
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool
from collections.abc import Generator

class GptImage1EditTool(Tool):
    def _invoke(
        self, tool_parameters: dict[str, Any]
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        invoke tools
        """
        azure_endpoint=self.runtime.credentials["azure_gpt_image_1_url"]
        api_version=self.runtime.credentials["azure_gpt_image_1_api_version"]
        api_key=self.runtime.credentials["azure_openai_api_key"]

        SIZE_MAPPING = {"square": "1024x1024", "vertical": "1024x1536", "horizontal": "1536x1024"}
        file = tool_parameters.get("upload_file")
        if not file:
            raise ValueError("PDF file is required")
        # print(f"file: {file}")

        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("Please input prompt")
        size = SIZE_MAPPING[tool_parameters.get("size", "square")]
        n = tool_parameters.get("n", 1)
        quality = tool_parameters.get("quality", "medium")
        if quality not in {"medium", "hd"}:
            yield self.create_text_message("Invalid quality")

        body = {
            "prompt": prompt,
            "n": n,
            "size": size,
            "quality": quality,
        }
        img = requests.get(f"http://192.168.1.134:82{file.url}")
        files = {
            "image": (file.filename, img.content, "image/png"),
        }
        response = requests.post(
            f"{azure_endpoint}edits?api-version={api_version}",
            headers={'Api-Key': api_key},
            data=body,
            files=files
        ).json()
        # print(response)
        for idx, item in enumerate(response['data']):
            b64_img = item['b64_json']
            if not b64_img:
                continue
            (mime_type, blob_image) = GptImage1EditTool._decode_image(b64_img)
            yield self.create_blob_message(
                blob=blob_image, meta={"mime_type": mime_type}
            )


    @staticmethod
    def _decode_image(base64_image: str) -> tuple[str, bytes]:
        """
        Decode a base64 encoded image. If the image is not prefixed with a MIME type,
        it assumes 'image/png' as the default.

        :param base64_image: Base64 encoded image string
        :return: A tuple containing the MIME type and the decoded image bytes
        """
        if GptImage1EditTool._is_plain_base64(base64_image):
            return ("image/png", base64.b64decode(base64_image))
        else:
            return GptImage1EditTool._extract_mime_and_data(base64_image)

    @staticmethod
    def _is_plain_base64(encoded_str: str) -> bool:
        """
        Check if the given encoded string is plain base64 without a MIME type prefix.

        :param encoded_str: Base64 encoded image string
        :return: True if the string is plain base64, False otherwise
        """
        return not encoded_str.startswith("data:image")

    @staticmethod
    def _extract_mime_and_data(encoded_str: str) -> tuple[str, bytes]:
        """
        Extract MIME type and image data from a base64 encoded string with a MIME type prefix.

        :param encoded_str: Base64 encoded image string with MIME type prefix
        :return: A tuple containing the MIME type and the decoded image bytes
        """
        mime_type = encoded_str.split(";")[0].split(":")[1]
        image_data_base64 = encoded_str.split(",")[1]
        decoded_data = base64.b64decode(image_data_base64)
        return (mime_type, decoded_data)
