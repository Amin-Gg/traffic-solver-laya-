# Traffic Solver — Jev 🚦

**[English](#english) · [فارسی](#فارسی)**

A 3D city whose traffic lights are controlled by **[Jev](https://docs.typesafe.ai/introduction)**, a structured decision model served by TypeSafe. Every decision is explained on screen: what Jev saw, what it read, what it answered, and what the light did. The project also includes a standalone **Jev playground + tutorial page**.

![Live city](docs/live-city.png)

---

<a id="english"></a>
## English

### What's inside

| Page | URL | What it shows |
|---|---|---|
| **Live city** | `/` | 3×3 grid of intersections, individual cars (IDM car-following), ambulances, accidents, rush hour. The city checks all 9 lights roughly every 0.25 s. New states use the Jev API. Click an intersection to see its sensors, the exact sentence Jev read, its answer and confidence. |
| **Why speed matters** | `/#compare` | Two identical cities side by side. One uses Jev's decisions as received. The other adds 4 s of delay to the same decisions, uses a fixed-time light, or uses a real LLM. |
| **How Jev decides** | `/#how` | Pipeline explanation, sliders that query the real model, and a 24-case benchmark. |
| **Jev playground & tutorial** | `/jev` | Standalone page: try any text/JSON with your own `choice` / `score` / `noul` questions, get generated Python/HTTP code, plus an install & usage guide. |

![Original playground layout](docs/laya-playground.png)

### How it works

```
browser (60 fps)                               server (FastAPI)
sim.js     cars, lanes, signals, sensors  ──►  brains.py: sensors → pressure → short English sentence
city3d.js  Three.js renderer                   Jev choice API / cached answers
app.js     panels, compare, playground    ◄──  "ns" / "ew" + probabilities
```

- The simulation runs in the browser, so motion is smooth.
- Jev only sets the *desired* phase. A safety layer does the actual switching: minimum green 12 s, yellow 2.5 s, all-red 1.2 s, and minimum green 1.5 s for an ambulance.
- Each intersection is summarised as a short sentence, e.g. *"North-south traffic is heavy. East-west traffic is light. More cars are waiting on the north-south road. An ambulance is coming on the east-west road."*
- There are only 105 possible sentences. The server caches each successful answer for the session. An unseen sentence gets its own TypeSafe request. Set `JEV_PREWARM=1` to fill the cache in the background at startup.

Historical measurements from the original Laya implementation:

| Input given to Laya | Result |
|---|---|
| Whole city in one big JSON | over the 512-token limit; answered "emergency" for every intersection |
| Raw numbers | ~50–75 % |
| Qualitative words only | ~75 % (biased toward the road named first) |
| Words + explicit comparison (used here) | 75/75 combinations, including 25 cases where the ambulance must override the busier road |

In offline simulations (3 seeds, 5 min each), ambulances waited ~7 s at red lights with 0.3 s decision latency, ~15 s with 4 s, ~23 s with 8 s, and 26–32 s with a fixed-time light.

### Setup

**Requirements:** Python 3.10+, internet access and a [TypeSafe API key](https://console.typesafe.ai/).

```bash
git clone https://github.com/YOUR-USERNAME/traffic-solver-laya-.git
cd traffic-solver-laya-

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
# macOS / Linux: export TYPESAFE_API_KEY="your-key"
# PowerShell:    $env:TYPESAFE_API_KEY = "your-key"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** (city) and **http://127.0.0.1:8000/jev** (playground). The status pill shows the Jev connection state. If a Jev answer is unavailable, the city uses its reference rule and labels the source `reference-fallback`.

**Docker:**

```bash
docker compose up --build -d      # then open http://localhost:8000
```

For Docker, copy `.env.example` to `.env`, then set `TYPESAFE_API_KEY` before starting Compose.

**Optional LLM comparison:** click ⚙ and set Ollama (`http://localhost:11434`, or `http://host.docker.internal:11434` inside Docker) or any OpenAI-compatible `/v1/chat/completions` endpoint. The LLM sees the same sentences and must return `ns` or `ew`.

**Tests:**

```bash
python -m pytest -q
```

### Project layout

```
app/main.py          FastAPI routes (/api/decide, /api/jev/predict, /api/benchmark, …)
app/brains.py        sensor fusion, sentences, Jev / LLM brains, answer cache
app/static/sim.js    traffic simulation (cars, signals, sensors, scenarios)
app/static/city3d.js Three.js city renderer (Three.js is vendored in static/vendor)
app/static/app.js    dashboard, compare view
app/static/laya.*    standalone playground & tutorial page
tests/               pytest suite
```

### Notes

- The page loads the Vazirmatn font from Google Fonts. Everything else, including Three.js, is served locally.
- Jev is served by TypeSafe. Legacy API paths containing `laya` also use Jev.

---

<a id="فارسی"></a>
<div dir="rtl">

## فارسی

### این پروژه چیست؟

یک شهر سه‌بعدی که چراغ‌های راهنمایی‌اش را **Jev** کنترل می‌کند. Jev یک مدل تصمیم‌گیری سریع است که متن تولید نمی‌کند. هر تصمیم روی صفحه توضیح داده می‌شود: Jev چه دید، چه جمله‌ای خواند، چه جوابی داد و چراغ چه شد. یک صفحه‌ی مستقل **آزمایشگاه و آموزش Jev** هم دارد.

| صفحه | آدرس | چه چیزی نشان می‌دهد |
|---|---|---|
| **شهر زنده** | `/` | ۹ چهارراه، خودروهای مستقل، آمبولانس، تصادف و ساعت شلوغی. شهر تقریباً هر ۰٫۲۵ ثانیه وضعیت هر ۹ چراغ را بررسی می‌کند و وضعیت‌های تازه را به Jev می‌فرستد. با کلیک روی هر چهارراه، حسگرها، جمله‌ی دقیقی که Jev خواند و جوابش را می‌بینید. |
| **چرا سرعت مهم است؟** | `/#compare` | دو شهر کاملاً یکسان کنار هم: شهر اول تصمیم Jev را بعد از دریافت اجرا می‌کند. برای شهر دوم می‌توان ۴ ثانیه تأخیر اضافه، چراغ زمان‌ثابت یا LLM واقعی انتخاب کرد. |
| **Jev چطور تصمیم می‌گیرد؟** | `/#how` | توضیح مراحل، اسلایدرهایی که از مدل واقعی می‌پرسند، و یک آزمون ۲۴ حالته. |
| **آزمایشگاه و آموزش Jev** | `/jev` | صفحه‌ی مستقل: هر متن یا JSON را با سؤال‌های دلخواه (`choice` / `score` / `noul`) امتحان کنید، کد Python/HTTP همان درخواست را بگیرید، و آموزش نصب و استفاده را بخوانید. |

### چطور کار می‌کند؟

- شبیه‌سازی در مرورگر اجرا می‌شود، پس حرکت خودروها روان است.
- سرور فقط «مغز» است: وضعیت هر چهارراه را به یک جمله‌ی کوتاه انگلیسی تبدیل می‌کند و از Jev می‌پرسد سبز با کدام خیابان باشد.
- Jev فقط «فاز مطلوب» را تعیین می‌کند. عوض کردن واقعی چراغ با قانون ایمنی انجام می‌شود: حداقل سبز ۱۲ ثانیه، زرد ۲٫۵ ثانیه و قرمز کامل ۱٫۲ ثانیه. برای آمبولانس حداقل سبز ۱٫۵ ثانیه است.
- کل جمله‌های ممکن فقط ۱۰۵ تاست. پاسخ موفق هر جمله تا پایان اجرای سرور در حافظه می‌ماند. برای پیش‌محاسبه در زمان شروع می‌توان `JEV_PREWARM=1` را فعال کرد.
- در شبیه‌سازی آفلاین (۳ شهر، هر کدام ۵ دقیقه):

  | تأخیر تصمیم | توقف آمبولانس پشت چراغ |
  |---|---|
  | ۰٫۳ ثانیه | ~۷ ثانیه |
  | ۴ ثانیه | ~۱۵ ثانیه |
  | ۸ ثانیه | ~۲۳ ثانیه |
  | چراغ زمان‌ثابت | ۲۶ تا ۳۲ ثانیه |

### راه‌اندازی

**پیش‌نیاز:** پایتون ۳٫۱۰ یا جدیدتر، اینترنت و [کلید API TypeSafe](https://console.typesafe.ai/).

</div>

```bash
git clone https://github.com/YOUR-USERNAME/traffic-solver-laya-.git
cd traffic-solver-laya-

python -m venv .venv
# Windows:        .venv\Scripts\activate
# macOS / Linux:  source .venv/bin/activate

pip install -r requirements.txt
# macOS / Linux: export TYPESAFE_API_KEY="your-key"
# PowerShell:    $env:TYPESAFE_API_KEY = "your-key"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

<div dir="rtl">

سپس این دو آدرس را باز کنید:
- **http://127.0.0.1:8000**: شهر
- **http://127.0.0.1:8000/jev**: آزمایشگاه

نشانگر بالای صفحه وضعیت اتصال Jev را نشان می‌دهد. اگر پاسخی از Jev در دسترس نباشد، شهر از قاعده‌ی مرجع استفاده می‌کند و منبع تصمیم با برچسب `reference-fallback` مشخص می‌شود.

**با Docker:** از فایل `.env.example` یک کپی با نام `.env` بسازید و `TYPESAFE_API_KEY` را وارد کنید.

</div>

```bash
docker compose up --build -d      # سپس http://localhost:8000
```

<div dir="rtl">

**مقایسه با LLM (اختیاری):** روی ⚙ بزنید و یکی از این دو را تنظیم کنید:
- Ollama: آدرس `http://localhost:11434`، یا داخل Docker آدرس `http://host.docker.internal:11434`
- هر سرویس سازگار با OpenAI (`/v1/chat/completions`)

LLM همان جمله‌هایی را می‌بیند که Jev می‌بیند و باید `ns` یا `ew` برگرداند.

**اجرای تست‌ها:** `python -m pytest -q`

**نکته‌ها:**
- اگر حافظه‌ی سیستم کم است، قبل از اجرا برنامه‌ها و تب‌های سنگین را ببندید.
- فونت Vazirmatn از Google Fonts بارگذاری می‌شود. بقیه‌ی فایل‌ها، از جمله Three.js، محلی‌اند.
- Jev را TypeSafe ارائه می‌کند. آدرس‌های قدیمی دارای `laya` همچنان باز می‌شوند و از Jev استفاده می‌کنند.

</div>
