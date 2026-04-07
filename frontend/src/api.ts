export type BoundingBox = {
	x: number;
	y: number;
	width: number;
	height: number;
};

export type Control = {
	id: string;
	type: string;
	label: string | null;
	bounding_box: BoundingBox;
	confidence: number;
};

export type DeviceContext = {
	device_type: string;
	brand: string | null;
	model: string | null;
	detected_controls: Control[];
	safety_warnings: string[];
};

export type DeviceAnalysisResult = {
	device_type: string;
	confidence: number;
	brand: string | null;
	model: string | null;
	suggested_categories: string[];
	detected_controls: Control[];
};

export type Step = {
	number: number;
	description: string;
	warning: string | null;
	highlighted_area: BoundingBox | null;
	completed: boolean;
};

export type Explanation = {
	text: string;
	steps: Step[] | null;
	warnings: string[];
	confidence: number;
	sources: string[];
};

type InstructionsResponse = {
	steps: Step[];
};

type ClarifyResponse = {
	text: string;
};

async function parseResponse<T>(response: Response): Promise<T> {
	if (!response.ok) {
		const payload = (await response.json().catch(() => null)) as {
			detail?: string;
		} | null;
		throw new Error(payload?.detail ?? "Request failed");
	}

	return (await response.json()) as T;
}

export async function analyzeImage(file: File): Promise<DeviceAnalysisResult> {
	const formData = new FormData();
	formData.append("file", file);

	const response = await fetch("/api/analyze", {
		method: "POST",
		body: formData,
	});

	return parseResponse<DeviceAnalysisResult>(response);
}

export async function explainDevice(input: {
	question: string;
	deviceContext: DeviceContext;
	imageBase64?: string;
	language?: string;
}): Promise<Explanation> {
	const response = await fetch("/api/explain", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			question: input.question,
			device_context: input.deviceContext,
			image_base64: input.imageBase64,
			language: input.language ?? "ru",
		}),
	});

	return parseResponse<Explanation>(response);
}

export async function generateInstructions(input: {
	task: string;
	deviceContext: DeviceContext;
	imageBase64?: string;
	language?: string;
}): Promise<Step[]> {
	const response = await fetch("/api/instructions", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			task: input.task,
			device_context: input.deviceContext,
			image_base64: input.imageBase64,
			language: input.language ?? "ru",
		}),
	});

	const payload = await parseResponse<InstructionsResponse>(response);
	return payload.steps;
}

export async function clarifyInstruction(input: {
	step: Step;
	question: string;
	deviceContext: DeviceContext;
	imageBase64?: string;
	language?: string;
}): Promise<string> {
	const response = await fetch("/api/clarify", {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			step: input.step,
			question: input.question,
			device_context: input.deviceContext,
			image_base64: input.imageBase64,
			language: input.language ?? "ru",
		}),
	});

	const payload = await parseResponse<ClarifyResponse>(response);
	return payload.text;
}

export async function fileToDataUrl(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();

		reader.onload = () => {
			if (typeof reader.result !== "string") {
				reject(new Error("Failed to read file"));
				return;
			}

			resolve(reader.result);
		};

		reader.onerror = () => {
			reject(reader.error ?? new Error("Failed to read file"));
		};

		reader.readAsDataURL(file);
	});
}
