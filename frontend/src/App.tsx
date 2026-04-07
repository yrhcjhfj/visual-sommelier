import { type FormEvent, useMemo, useState } from "react";

import {
	analyzeImage,
	type DeviceAnalysisResult,
	type DeviceContext,
	explainDevice,
	fileToDataUrl,
	generateInstructions,
	type Step,
} from "./api";

const defaultQuestion =
	"Что означает эта панель управления и как ей пользоваться?";
const defaultTask = "Подскажи пошагово, как безопасно запустить устройство.";

function formatConfidence(confidence: number) {
	return `${Math.round(confidence * 100)}%`;
}

function toDeviceContext(
	result: DeviceAnalysisResult | null,
): DeviceContext | null {
	if (!result) {
		return null;
	}

	return {
		device_type: result.device_type,
		brand: result.brand,
		model: result.model,
		detected_controls: result.detected_controls,
		safety_warnings: [],
	};
}

function App() {
	const [selectedFile, setSelectedFile] = useState<File | null>(null);
	const [imagePreview, setImagePreview] = useState<string | null>(null);
	const [analysis, setAnalysis] = useState<DeviceAnalysisResult | null>(null);
	const [question, setQuestion] = useState(defaultQuestion);
	const [task, setTask] = useState(defaultTask);
	const [explanationText, setExplanationText] = useState<string>("");
	const [explanationWarnings, setExplanationWarnings] = useState<string[]>([]);
	const [instructionSteps, setInstructionSteps] = useState<Step[]>([]);
	const [status, setStatus] = useState<string>(
		"Загрузите изображение устройства для анализа.",
	);
	const [error, setError] = useState<string>("");
	const [isAnalyzing, setIsAnalyzing] = useState(false);
	const [isExplaining, setIsExplaining] = useState(false);
	const [isGeneratingInstructions, setIsGeneratingInstructions] =
		useState(false);

	const deviceContext = useMemo(() => toDeviceContext(analysis), [analysis]);

	async function handleFileChange(event: FormEvent<HTMLInputElement>) {
		const file = event.currentTarget.files?.[0] ?? null;

		setSelectedFile(file);
		setAnalysis(null);
		setExplanationText("");
		setExplanationWarnings([]);
		setInstructionSteps([]);
		setError("");

		if (!file) {
			setImagePreview(null);
			setStatus("Загрузите изображение устройства для анализа.");
			return;
		}

		try {
			const preview = await fileToDataUrl(file);
			setImagePreview(preview);
			setStatus("Изображение загружено. Теперь можно запустить анализ.");
		} catch (fileError) {
			setImagePreview(null);
			setError(
				fileError instanceof Error
					? fileError.message
					: "Не удалось прочитать файл",
			);
		}
	}

	async function handleAnalyze() {
		if (!selectedFile) {
			setError("Сначала выберите изображение устройства.");
			return;
		}

		setIsAnalyzing(true);
		setError("");
		setStatus("Анализируем изображение устройства...");

		try {
			const result = await analyzeImage(selectedFile);
			setAnalysis(result);
			setStatus(
				"Анализ завершен. Теперь можно запросить объяснение или инструкции.",
			);
		} catch (requestError) {
			setError(
				requestError instanceof Error
					? requestError.message
					: "Ошибка анализа изображения",
			);
			setStatus("Анализ не завершен.");
		} finally {
			setIsAnalyzing(false);
		}
	}

	async function handleExplain() {
		if (!selectedFile || !deviceContext) {
			setError("Сначала выполните анализ устройства.");
			return;
		}

		setIsExplaining(true);
		setError("");
		setStatus("Генерируем объяснение...");

		try {
			const imageBase64 = await fileToDataUrl(selectedFile);
			const result = await explainDevice({
				question,
				deviceContext,
				imageBase64,
			});

			setExplanationText(result.text);
			setExplanationWarnings(result.warnings);
			setStatus("Объяснение готово.");
		} catch (requestError) {
			setError(
				requestError instanceof Error
					? requestError.message
					: "Ошибка генерации объяснения",
			);
			setStatus("Не удалось получить объяснение.");
		} finally {
			setIsExplaining(false);
		}
	}

	async function handleGenerateInstructions() {
		if (!selectedFile || !deviceContext) {
			setError("Сначала выполните анализ устройства.");
			return;
		}

		setIsGeneratingInstructions(true);
		setError("");
		setStatus("Генерируем пошаговые инструкции...");

		try {
			const imageBase64 = await fileToDataUrl(selectedFile);
			const steps = await generateInstructions({
				task,
				deviceContext,
				imageBase64,
			});

			setInstructionSteps(steps);
			setStatus("Инструкции готовы.");
		} catch (requestError) {
			setError(
				requestError instanceof Error
					? requestError.message
					: "Ошибка генерации инструкций",
			);
			setStatus("Не удалось получить инструкции.");
		} finally {
			setIsGeneratingInstructions(false);
		}
	}

	return (
		<div className="min-h-screen bg-stone-100 text-stone-900 transition-colors dark:bg-stone-950 dark:text-stone-100">
			<div className="mx-auto flex min-h-screen max-w-7xl flex-col gap-6 px-4 py-6 sm:px-6 lg:px-8">
				<header className="rounded-[2rem] border border-stone-200 bg-white/90 p-6 shadow-sm backdrop-blur dark:border-stone-800 dark:bg-stone-900/90">
					<div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
						<div className="max-w-3xl">
							<p className="text-sm uppercase tracking-[0.3em] text-amber-600 dark:text-amber-400">
								Local device assistant
							</p>
							<h1 className="mt-2 text-4xl font-semibold tracking-tight sm:text-5xl">
								Визуальный Сомелье
							</h1>
							<p className="mt-3 max-w-2xl text-sm text-stone-600 dark:text-stone-300 sm:text-base">
								Загрузите фотографию панели управления, получите распознавание
								устройства и сразу запросите объяснение или безопасные пошаговые
								действия через backend API.
							</p>
						</div>

						<div className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/60 dark:text-amber-100">
							<div className="font-medium">Статус</div>
							<div className="mt-1 max-w-sm text-amber-800 dark:text-amber-200">
								{status}
							</div>
						</div>
					</div>
				</header>

				<main className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
					<section className="rounded-[2rem] border border-stone-200 bg-white p-5 shadow-sm dark:border-stone-800 dark:bg-stone-900">
						<div className="flex items-center justify-between gap-3">
							<div>
								<h2 className="text-xl font-semibold">Снимок устройства</h2>
								<p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
									Поддерживается локальная загрузка изображения и отправка на
									`/api/analyze`.
								</p>
							</div>
						</div>

						<label className="mt-5 flex cursor-pointer flex-col items-center justify-center rounded-[1.5rem] border border-dashed border-stone-300 bg-stone-50 px-6 py-10 text-center transition hover:border-amber-400 hover:bg-amber-50/60 dark:border-stone-700 dark:bg-stone-950 dark:hover:border-amber-500 dark:hover:bg-amber-950/30">
							<input
								className="hidden"
								type="file"
								accept="image/*"
								onChange={handleFileChange}
							/>
							<span className="text-sm font-medium text-stone-700 dark:text-stone-200">
								{selectedFile ? "Заменить изображение" : "Выбрать изображение"}
							</span>
							<span className="mt-2 text-xs text-stone-500 dark:text-stone-400">
								JPG, PNG или другое `image/*`
							</span>
						</label>

						<div className="mt-5 grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
							<div className="overflow-hidden rounded-[1.5rem] bg-stone-100 dark:bg-stone-950">
								{imagePreview ? (
									<img
										alt="Предпросмотр устройства"
										className="h-full min-h-[280px] w-full object-cover"
										src={imagePreview}
									/>
								) : (
									<div className="flex min-h-[280px] items-center justify-center px-6 text-center text-sm text-stone-500 dark:text-stone-400">
										Предпросмотр появится после выбора изображения.
									</div>
								)}
							</div>

							<div className="flex flex-col gap-4 rounded-[1.5rem] border border-stone-200 bg-stone-50 p-4 dark:border-stone-800 dark:bg-stone-950/70">
								<div>
									<div className="text-sm font-medium text-stone-700 dark:text-stone-200">
										Файл
									</div>
									<div className="mt-1 text-sm text-stone-500 dark:text-stone-400">
										{selectedFile
											? `${selectedFile.name} · ${Math.round(selectedFile.size / 1024)} KB`
											: "Файл не выбран"}
									</div>
								</div>

								<button
									className="inline-flex items-center justify-center rounded-full bg-stone-900 px-5 py-3 text-sm font-medium text-white transition hover:bg-stone-700 disabled:cursor-not-allowed disabled:bg-stone-400 dark:bg-stone-100 dark:text-stone-900 dark:hover:bg-stone-300 dark:disabled:bg-stone-700 dark:disabled:text-stone-400"
									disabled={!selectedFile || isAnalyzing}
									onClick={handleAnalyze}
									type="button"
								>
									{isAnalyzing ? "Идет анализ..." : "Запустить анализ"}
								</button>

								{analysis ? (
									<div className="rounded-[1.25rem] border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-950 dark:border-emerald-900/70 dark:bg-emerald-950/40 dark:text-emerald-100">
										<div className="text-xs uppercase tracking-[0.2em] text-emerald-700 dark:text-emerald-300">
											Результат анализа
										</div>
										<div className="mt-3 text-lg font-semibold capitalize">
											{analysis.device_type.replaceAll("_", " ")}
										</div>
										<div className="mt-2 text-sm">
											Уверенность: {formatConfidence(analysis.confidence)}
										</div>
										<div className="mt-1 text-sm text-emerald-800 dark:text-emerald-200">
											Найдено элементов управления:{" "}
											{analysis.detected_controls.length}
										</div>
									</div>
								) : null}

								{error ? (
									<div className="rounded-[1.25rem] border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900 dark:border-rose-900/60 dark:bg-rose-950/40 dark:text-rose-100">
										{error}
									</div>
								) : null}
							</div>
						</div>
					</section>

					<section className="rounded-[2rem] border border-stone-200 bg-white p-5 shadow-sm dark:border-stone-800 dark:bg-stone-900">
						<h2 className="text-xl font-semibold">Распознанный контекст</h2>
						<div className="mt-4 flex flex-wrap gap-2">
							{analysis?.suggested_categories.length ? (
								analysis.suggested_categories.map((category) => (
									<span
										key={category}
										className="rounded-full border border-stone-300 px-3 py-1 text-xs uppercase tracking-[0.18em] text-stone-600 dark:border-stone-700 dark:text-stone-300"
									>
										{category.replaceAll("_", " ")}
									</span>
								))
							) : (
								<span className="text-sm text-stone-500 dark:text-stone-400">
									После анализа здесь появятся suggested categories и элементы
									управления.
								</span>
							)}
						</div>

						<div className="mt-5 space-y-3">
							{analysis?.detected_controls.map((control) => (
								<div
									key={control.id}
									className="rounded-[1.25rem] border border-stone-200 bg-stone-50 p-4 dark:border-stone-800 dark:bg-stone-950/70"
								>
									<div className="flex items-start justify-between gap-3">
										<div>
											<div className="text-sm font-medium capitalize text-stone-900 dark:text-stone-100">
												{control.type.replaceAll("_", " ")}
											</div>
											<div className="mt-1 text-sm text-stone-500 dark:text-stone-400">
												{control.label || "Без текстовой метки"}
											</div>
										</div>
										<span className="rounded-full bg-stone-900 px-3 py-1 text-xs text-white dark:bg-stone-100 dark:text-stone-900">
											{formatConfidence(control.confidence)}
										</span>
									</div>
								</div>
							))}
						</div>
					</section>

					<section className="rounded-[2rem] border border-stone-200 bg-white p-5 shadow-sm dark:border-stone-800 dark:bg-stone-900">
						<div className="flex items-center justify-between gap-3">
							<div>
								<h2 className="text-xl font-semibold">Объяснение</h2>
								<p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
									Вопрос отправляется в `/api/explain` вместе с `device_context`
									и изображением.
								</p>
							</div>
							<button
								className="inline-flex items-center justify-center rounded-full bg-amber-500 px-4 py-2 text-sm font-medium text-stone-950 transition hover:bg-amber-400 disabled:cursor-not-allowed disabled:bg-stone-300 dark:disabled:bg-stone-700 dark:disabled:text-stone-400"
								disabled={!deviceContext || isExplaining}
								onClick={handleExplain}
								type="button"
							>
								{isExplaining ? "Генерация..." : "Получить объяснение"}
							</button>
						</div>

						<label className="mt-4 block text-sm font-medium text-stone-700 dark:text-stone-200">
							Вопрос
							<textarea
								className="mt-2 min-h-[132px] w-full rounded-[1.25rem] border border-stone-300 bg-stone-50 px-4 py-3 text-sm text-stone-900 outline-none transition focus:border-amber-500 focus:ring-2 focus:ring-amber-200 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-amber-900"
								onChange={(event) => setQuestion(event.target.value)}
								value={question}
							/>
						</label>

						{explanationWarnings.length ? (
							<div className="mt-4 rounded-[1.5rem] border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-100">
								<div className="font-medium">Предупреждения</div>
								<div className="mt-2 space-y-2">
									{explanationWarnings.map((warning) => (
										<p key={warning}>{warning}</p>
									))}
								</div>
							</div>
						) : null}

						<div className="mt-4 rounded-[1.5rem] border border-stone-200 bg-stone-50 p-4 text-sm leading-6 text-stone-700 dark:border-stone-800 dark:bg-stone-950/70 dark:text-stone-200">
							{explanationText ||
								"После запроса здесь появится объяснение от LLaVA через backend."}
						</div>
					</section>

					<section className="rounded-[2rem] border border-stone-200 bg-white p-5 shadow-sm dark:border-stone-800 dark:bg-stone-900">
						<div className="flex items-center justify-between gap-3">
							<div>
								<h2 className="text-xl font-semibold">Пошаговые инструкции</h2>
								<p className="mt-1 text-sm text-stone-500 dark:text-stone-400">
									Задача отправляется в `/api/instructions` и возвращает
									структурированные шаги.
								</p>
							</div>
							<button
								className="inline-flex items-center justify-center rounded-full border border-stone-300 px-4 py-2 text-sm font-medium text-stone-900 transition hover:border-stone-400 hover:bg-stone-100 disabled:cursor-not-allowed disabled:border-stone-200 disabled:text-stone-400 dark:border-stone-700 dark:text-stone-100 dark:hover:border-stone-500 dark:hover:bg-stone-800 dark:disabled:border-stone-800 dark:disabled:text-stone-600"
								disabled={!deviceContext || isGeneratingInstructions}
								onClick={handleGenerateInstructions}
								type="button"
							>
								{isGeneratingInstructions ? "Генерация..." : "Получить шаги"}
							</button>
						</div>

						<label className="mt-4 block text-sm font-medium text-stone-700 dark:text-stone-200">
							Задача
							<textarea
								className="mt-2 min-h-[120px] w-full rounded-[1.25rem] border border-stone-300 bg-stone-50 px-4 py-3 text-sm text-stone-900 outline-none transition focus:border-stone-500 focus:ring-2 focus:ring-stone-200 dark:border-stone-700 dark:bg-stone-950 dark:text-stone-100 dark:focus:ring-stone-800"
								onChange={(event) => setTask(event.target.value)}
								value={task}
							/>
						</label>

						<div className="mt-4 space-y-3">
							{instructionSteps.length ? (
								instructionSteps.map((step) => (
									<article
										key={`${step.number}-${step.description}`}
										className="rounded-[1.5rem] border border-stone-200 bg-stone-50 p-4 dark:border-stone-800 dark:bg-stone-950/70"
									>
										<div className="text-xs uppercase tracking-[0.18em] text-stone-500 dark:text-stone-400">
											Шаг {step.number}
										</div>
										<p className="mt-2 text-sm leading-6 text-stone-800 dark:text-stone-200">
											{step.description}
										</p>
										{step.warning ? (
											<p className="mt-3 rounded-2xl border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-900/60 dark:bg-amber-950/40 dark:text-amber-100">
												{step.warning}
											</p>
										) : null}
									</article>
								))
							) : (
								<div className="rounded-[1.5rem] border border-dashed border-stone-300 px-4 py-8 text-center text-sm text-stone-500 dark:border-stone-700 dark:text-stone-400">
									После запроса здесь появятся пошаговые инструкции.
								</div>
							)}
						</div>
					</section>
				</main>
			</div>
		</div>
	);
}

export default App;
