# Примеры

Эти руководства описывают три plugin-проекта из examples/. Это рабочие проекты, а не минимальные заглушки. Установите plugin и подготовьте внешние данные и weights, указанные на его странице.

| Проект | Задача и входы | Training | Evaluation | Inference | Внешние требования | Основные понятия |
| --- | --- | --- | --- | --- | --- | --- |
| VA estimation | Valence/arousal по audio или frame features | Audio и multimodal configs | Audio и multimodal configs | Конфигурации нет | VA annotations, feature cache, plugin checkpoint | Dataset, framewise callbacks, plugin loss/metrics |
| ORAGEN | Gender и age по audio/video; mask labels optional | Audio и multimodal configs | Audio и multimodal configs | Есть video pipeline | Corpora, VAD, .dat cache, remote ORAGEN weights | Plugin models, checkpoint resolution, window aggregation |
| Affective states | Emotion и sentiment по audio, video и transcript | Multimodal config | Multimodal config | Есть video pipeline | CMU-MOSEI/MELD/RAMAS, Whisper, YOLO, HF weights | Multi-task output, parallel DAG, feature cache |

Обучение и evaluation требуют внешних datasets, features и annotations. Sample media у ORAGEN и affective-states не означают, что weights и runtime dependencies уже установлены. Для первого framework-пути используйте [Первый эксперимент](../getting-started/first-experiment.md).
