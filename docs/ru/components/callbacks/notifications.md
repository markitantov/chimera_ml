# Колбэк уведомлений Telegram — `TelegramNotifierCallback`

`TelegramNotifierCallback`, key `telegram_notifier_callback`, создаёт session в `on_fit_start`, отслеживает best monitor в `on_epoch_end` и отправляет сообщение в `on_fit_end`.

| Parameter | Default |
| --- | --- |
| `token_env_field` / `chat_id_env_field` | `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` |
| `message_prefix` / `message_suffix` | `✅ Training finished` / `""` |
| `include_mlflow_experiment_name` | `True` |
| `include_trainer_progress` | `True` |
| `include_last_logs` / `include_best_logs` | `True` |
| `monitor` / `mode` / `min_delta` | `val/loss` / `min` / `0.0` |
| `request_timeout_sec` / `telegram_parse_mode` | `15` / `HTML` |

Требуются environment variables и dependency `requests`; HTTP failures логируются как warning. Никогда не записывайте tokens в YAML.

```bash
export TELEGRAM_BOT_TOKEN='your-token'
export TELEGRAM_CHAT_ID='your-chat-id'
```

```yaml
- name: telegram_notifier_callback
  params: {monitor: val/mae, mode: min, request_timeout_sec: 15}
```
