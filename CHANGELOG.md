# Changelog

## Unreleased

### Added
- Updates without dropping calls: `compose.bluegreen.yaml` (Caddy in front of `runtime-blue` / `runtime-green`)
  and `scripts/deploy.sh`. A new instance takes the bundles over from the running one (`VA_PEER_URL`,
  `GET /internal/handoff`, memory to memory) and warms the greetings before `GET /ready` answers 200; the old
  one is drained (`POST /internal/drain`, `voice-agent-runtime drain`) while results and acknowledgements of
  its calls are forwarded from the new one. `/health` gives the PBX `boot_id` and `capabilities`;
  `voice-agent-runtime status` and `healthcheck --ready`. Version 0.2.0.
- Silero TTS voices (v5_cis_base, MIT, stress by silero-stress): `silero-ekaterina`, `silero-oksana`,
  `silero-zinaida`, `silero-roman`, `silero-eduard`, 3-4 times faster than Piper. `VA_VOICES` selects the loaded
  voices; `GET /v1/voices` and `GET /v1/voices/{id}/sample` (WAV) for the module voice picker and preview.
  A bundle with a voice the runtime does not load speaks with `VA_DEFAULT_VOICE` instead of being rejected.
- Incremental speech recognition of long turns (`incremental.py`): each speculation recognizes only the audio
  after the last internal pause of at least `VA_STT_CHUNK_PAUSE_MS` (160); such pieces are committed while the
  caller keeps talking, so the end of the turn waits for the last phrase only and every second of audio is
  recognized about once. `VA_STT_CHUNKS`, `VA_STT_CHUNK_PAUSE_MS`, `VA_STT_CHUNK_MIN_MS`. New turn-detector
  event `CHUNK`; speculation, turn end and barge-in timing are unchanged.
- Hotwords: optional bundle field `hotwords` (list of strings, at most 200, each at most 100 characters) plus
  terms derived from the bundle bias GigaAM's greedy transducer decoder (`biasing.py`, the LocalSTTWorker
  approach, passed per call through onnx-asr kwargs so concurrent calls do not share state) and restore
  abbreviations that lost their first letter ("ТС" -> "АТС"). `VA_HOTWORDS`, `VA_HOTWORD_BOOST`.
- Usage metering (`usage.py`): per day, tenant and agent - calls, connected and agent speaking seconds, caller
  turns, LLM requests and input/output tokens (from `stream_options.include_usage`, estimated otherwise), TTS
  characters, fallbacks. Aggregates only, in SQLite `VA_USAGE_DB` (`/data` volume), flushed every
  `VA_USAGE_FLUSH_SECONDS`. `GET /v1/usage?from=&to=` for the PBX, `voice_agent_usage_*_total{tenant}` in
  `/metrics`. `LLM_STREAM_USAGE` / `LLM2_STREAM_USAGE` turn the usage request off.

### Changed
- A hand-set stress mark ("+" before a vowel, "МИК+О") in the agent texts now reaches Silero instead of being
  turned into a space; Piper and recognition hints drop it.
- The agent ends the call on an explicit request ("заверши звонок", "сбрось", "кладу трубку", "до встречи"), not
  only on a farewell. Narration between single asterisks ("*Звонок завершён.*") is not said; a reply left empty
  asks what else the caller needs.
- A turn whose LLM request failed before anything was said is asked once more instead of handing the caller to a
  person at once (a provider 500 on a malformed model message ended a call).
- A callback to a short internal caller number ("333") is accepted; before, the agent kept asking for the phone.
- Echo of the agent from a caller's speakerphone no longer interrupts it or gets an answer: an interruption is
  confirmed by recognition (at most 0.8 s, then it interrupts as before), and caller speech that started while
  the agent was audible and is only the agent's recent words is ignored (`echo.py`, metrics `echo_ignored`,
  `barge_ins_ignored`). Before, the greeting was cut after "Здравствуйте" and the agent answered itself twice.
- Echo cancellation: the caller audio goes through WebRTC AEC3 (livekit, Apache-2.0; `VA_AEC`) with the agent
  audio as the reference before VAD and recognition, and speech over the agent that it took 6 dB or more off is
  echo. On three recorded speakerphone calls it told all 7 echo pieces (7.9-34.5 dB off) from all 4 caller
  phrases (0.1-0.8 dB off).
- An interruption is checked once per phrase on its own audio (0.6 s of it); before, every voiced hop over the
  agent re-recognized the call from its start and blocked recognition of the caller's next phrases.
- A transfer on a question about the options ("А какие отделы есть?", "С кем можешь соединить?") does not
  happen: the agent names the departments from the bundle queues. No second offer when the reply already asks;
  tool arguments printed as text are not said; the prompt forbids "соединяю" without a transfer and invented
  modules, tariffs and conditions.
- The call ends on "хватит уже", "уйди", "отстань", "всё, спасибо", and on "ничем" / "нет" / "ничего не надо"
  answering "Чем ещё могу помочь?" (not answering an offer such as "Соединить вас?"); before, the agent kept
  asking "Чем ещё могу помочь?" for minutes.
- A callback request with a name the caller never said ("Клиент") or a time when the caller named none (a
  made-up "сейчас") is not saved: the agent asks for it. "Обратный звонок" is not a topic. No second
  "Записал вашу заявку" when the model already confirmed. Special tokens and filler (`<|start|>assistant`,
  `(ожидаем)......`) are not said.
- The prompt keeps the agent in its role: requests to forget or change the rules, switch roles, retell the
  instructions or talk about unrelated things get one polite refusal and a question about the company's
  business; the company's own extra rules still apply.
- A clause that is an English reasoning note ("Proceed with request_callback using phone 333.") is not said.
- `SpeechEngines.recognize(audio, terms=())` takes the agent's terms; `Turn` receives the futures of its
  recognized pieces instead of the audio.
- compose: `data` volume at `/data`; Dockerfile declares it.
