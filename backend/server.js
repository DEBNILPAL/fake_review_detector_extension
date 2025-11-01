
import express from "express";
import cors from "cors";
import bodyParser from "body-parser";
import pkg from "pg";
import dotenv from "dotenv";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";

const {Pool} = pkg;

dotenv.config();

const app = express();
const port = Number(process.env.PORT) || 8000;

// Middleware
app.use(bodyParser.json());
app.use(cors());

let pool;
if (process.env.DATABASE_URL) {
  pool = new Pool({
    connectionString: process.env.DATABASE_URL,
    ssl: process.env.PGSSL === 'false' ? false : { rejectUnauthorized: false }
  });
} else {
  pool = new Pool({
    user: process.env.DB_USER,
    host: process.env.DB_HOST,
    database: process.env.DB_NAME,
    password: process.env.DB_PASSWORD,
    port: process.env.DB_PORT,
    ssl: process.env.PGSSL === 'true' ? { rejectUnauthorized: false } : false
  });
}

// Ensure DB schema exists
async function ensureSchema() {
  const ddl = `
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      full_name VARCHAR(200) NOT NULL,
      email VARCHAR(255) UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS reviews (
      id SERIAL PRIMARY KEY,
      user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
      client_name VARCHAR(200) NOT NULL,
      rating NUMERIC,
      review_text TEXT NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TABLE IF NOT EXISTS review_analysis (
      id SERIAL PRIMARY KEY,
      full_name VARCHAR(200) NOT NULL,
      email VARCHAR(255) NOT NULL,
      review TEXT NOT NULL,
      analysis json NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
    CREATE INDEX IF NOT EXISTS idx_reviews_user_id ON reviews(user_id);
    CREATE INDEX IF NOT EXISTS idx_review_analysis_email ON review_analysis(email);
    CREATE INDEX IF NOT EXISTS idx_review_analysis_created_at ON review_analysis(created_at);

    -- Predictions table used by Chrome Extension continuous learning and scoring history
    CREATE TABLE IF NOT EXISTS predict (
      id SERIAL PRIMARY KEY,
      review_text TEXT NOT NULL,
      rating NUMERIC,
      product_id TEXT,
      reviewer_id TEXT,
      prediction TEXT,
      prob_fake NUMERIC,
      features json,
      components json,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_predict_created_at ON predict(created_at);

    -- Migrations to relax constraints if table existed with narrower types
    DO $$ BEGIN
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='predict' AND column_name='product_id' AND data_type IN ('character varying','varchar')
      ) THEN
        EXECUTE 'ALTER TABLE predict ALTER COLUMN product_id TYPE TEXT';
      END IF;
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='predict' AND column_name='reviewer_id' AND data_type IN ('character varying','varchar')
      ) THEN
        EXECUTE 'ALTER TABLE predict ALTER COLUMN reviewer_id TYPE TEXT';
      END IF;
      -- Ensure features/components are json and nulls allowed
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='predict' AND column_name='features' AND data_type <> 'json'
      ) THEN
        EXECUTE 'ALTER TABLE predict ALTER COLUMN features TYPE json USING NULLIF(features, '''')::json';
      END IF;
      IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name='predict' AND column_name='components' AND data_type <> 'json'
      ) THEN
        EXECUTE 'ALTER TABLE predict ALTER COLUMN components TYPE json USING NULLIF(components, '''')::json';
      END IF;
    END $$;
  `;
  try {
    await pool.query(ddl);
    console.log("DB schema ensured.");
  } catch (e) {
    console.error("Failed to ensure schema:", e);
  }
}

// Deep Learning Inference Integration
const PYTHON_PATH = process.env.PYTHON_PATH || "python";
const INFERENCE_SCRIPT = path.resolve(process.cwd(), "..", "deep learning", "inference_service.py");

function runInference(command, payload = null) {
  return new Promise((resolve, reject) => {
    try {
      const args = [INFERENCE_SCRIPT, command];
      const proc = spawn(PYTHON_PATH, args, { stdio: ["pipe", "pipe", "pipe"] });

      let stdout = "";
      let stderr = "";

      proc.stdout.on("data", (d) => (stdout += d.toString()));
      proc.stderr.on("data", (d) => (stderr += d.toString()));

      proc.on("error", (err) => reject(err));
      proc.on("close", (code) => {
        if (code !== 0) {
          return reject(new Error(`Inference process exited with code ${code}: ${stderr}`));
        }
        try {
          const json = JSON.parse(stdout || "{}");
          if (json && json.error) {
            return reject(new Error(json.error));
          }
          resolve(json);
        } catch (e) {
          reject(new Error(`Failed to parse inference output: ${e}\nRaw: ${stdout}\nStderr: ${stderr}`));
        }
      });

      if (payload) {
        proc.stdin.write(JSON.stringify(payload));
      }
      proc.stdin.end();
    } catch (e) {
      reject(e);
    }
  });
}

app.post('/api/signup', async (req, res) => {
    const { username,email,password} = req.body;
    try {
        const result = await pool.query(
            'INSERT INTO users (full_name, email, password_hash) VALUES ($1, $2, $3) RETURNING *',
            [username, email, password]
        );
        res.status(201).json(result.rows[0]);
    } catch (error) {
        console.error('Error inserting data:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

// Fetch reviews from DB
app.get('/api/reviews', async (_req, res) => {
  try {
    const r = await pool.query('SELECT * FROM reviews ORDER BY created_at DESC LIMIT 500');
    res.json({ rows: r.rows, count: r.rowCount });
  } catch (e) {
    console.error('Error fetching reviews:', e);
    res.status(500).json({ error: 'Failed to fetch reviews' });
  }
});

// Diagnostics endpoint
app.get('/api/diagnostics', async (_req, res) => {
  const diag = { db: {}, python: {}, artifacts: {} };
  try {
    const dbver = await pool.query('SELECT version()');
    diag.db.ok = true;
    diag.db.version = dbver.rows?.[0]?.version;
    const tables = await pool.query(`
      SELECT table_name FROM information_schema.tables 
      WHERE table_schema='public' AND table_name IN ('users','reviews','review_analysis')
    `);
    diag.db.tables = tables.rows.map(r => r.table_name);
  } catch (e) {
    diag.db.ok = false;
    diag.db.error = String(e);
  }

  diag.python.path = PYTHON_PATH;
  diag.python.script = INFERENCE_SCRIPT;
  diag.python.script_exists = fs.existsSync(INFERENCE_SCRIPT);

  const base = path.resolve(process.cwd(), "..", "deep learning");
  const files = [
    "deep_learning_model.keras",
    "scaler.joblib",
    "gbc_model.joblib",
    "tfidf_vectorizer.joblib",
    "reviews_large.csv",
  ];
  for (const f of files) {
    const p = path.join(base, f);
    diag.artifacts[f] = fs.existsSync(p);
  }

  try {
    const analytics = await runInference('analytics');
    diag.sample_analytics = analytics?.total_rows !== undefined ? 'ok' : 'unknown';
  } catch (e) {
    diag.sample_analytics = `error: ${e}`;
  }

  res.json(diag);
});

app.post('/api/login', async (req, res) => {
    const { email, password } = req.body;
    try {
        const result = await pool.query(
            'SELECT * FROM users WHERE email = $1 AND password_hash = $2',
            [email, password]
        );
        if (result.rows.length > 0) {
            res.status(200).json({ message: 'Login successful', user: result.rows[0] });
        } else {
            res.status(401).json({ error: 'Invalid email or password' });
        }
    } catch (error) {
        console.error('Error querying data:', error);
        res.status(500).json({ error: 'Internal Server Error' });
    }   
});

app.get('/api/analytics', async (req, res) => {
  try {
    const result = await runInference('analytics');
    res.json(result);
  } catch (error) {
    console.error('Error fetching analytics via model:', error);
    res.status(500).json({ error: 'Failed to compute analytics from model dataset' });
  }
});

// New: POST /api/analytics to analyze a single review and persist to DB
app.post('/api/analytics', async (req, res) => {
  const { full_name, email, review, rating, productId, reviewerId } = req.body || {};

  if (!full_name || !email || !review) {
    return res.status(400).json({ error: 'Fields full_name, email, and review are required.' });
  }

  try {
    // Call ML predictor (uses same artifacts as /api/predict)
    const inference = await runInference('predict', {
      text: review,
      rating: rating ?? 0,
      productId: productId ?? 'prod-1',
      reviewerId: reviewerId ?? 'user-1',
    });

    // Persist into review_analysis table
    const insertQuery = `
      INSERT INTO review_analysis (full_name, email, review, analysis, created_at)
      VALUES ($1, $2, $3, $4::json, NOW())
      RETURNING id
    `;
    const values = [full_name, email, review, JSON.stringify(inference)];
    const saved = await pool.query(insertQuery, values);

    return res.status(201).json({
      message: 'Review analyzed and saved.',
      analysis: inference,
      id: saved.rows?.[0]?.id,
    });
  } catch (error) {
    console.error('Error during review analysis:', error);
    return res.status(500).json({ error: 'Failed to analyze and save review.' });
  }
});

app.post('/api/predict', async (req, res) => {
  try {
    const { reviews } = req.body || {};

    // ---------- Batch Mode ----------
    if (Array.isArray(reviews)) {
      const predictions = [];

      for (const r of reviews) {
        const { id, text, rating, url, domain } = r || {};
        if (!text) continue;

        const inf = await runInference('predict', { text, rating: rating ?? 0 });
        predictions.push({ id, prediction: inf.prediction, prob_fake: inf.prob_fake });

        try {
          // Convert objects safely to string for TEXT columns
          const featuresStr =
            inf?.features && typeof inf.features === 'object'
              ? JSON.stringify(inf.features)
              : inf?.features?.toString?.() || null;

          const componentsStr =
            inf?.components && typeof inf.components === 'object'
              ? JSON.stringify(inf.components)
              : inf?.components?.toString?.() || null;

          await pool.query(
            `INSERT INTO predict (
                review_text, rating, product_id, reviewer_id,
                prediction, prob_fake, features, components, created_at
             ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())`,
            [
              text,
              rating ?? null,
              domain ? String(domain).slice(0, 512) : null,
              url ? String(url).slice(0, 1024) : null,
              inf.prediction,
              inf.prob_fake,
              featuresStr,
              componentsStr
            ]
          );
        } catch (e) {
          console.warn('predict insert failed:', e.message);
        }
      }

      return res.json({ predictions });
    }

    // ---------- Single Review Mode ----------
    const { text, rating, productId, reviewerId } = req.body || {};
    if (!text || rating === undefined) {
      return res.status(400).json({ error: 'Fields text and rating are required.' });
    }

    const result = await runInference('predict', { text, rating, productId, reviewerId });
    const { prediction, prob_fake, features, components } = result || {};

    const featuresStr =
      features && typeof features === 'object'
        ? JSON.stringify(features)
        : features?.toString?.() || null;

    const componentsStr =
      components && typeof components === 'object'
        ? JSON.stringify(components)
        : components?.toString?.() || null;

    const saved = await pool.query(
      `INSERT INTO predict (
          review_text, rating, product_id, reviewer_id,
          prediction, prob_fake, features, components, created_at
       ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())
       RETURNING id`,
      [
        text,
        rating,
        (productId || 'unknown').toString().slice(0, 512),
        (reviewerId || 'unknown').toString().slice(0, 1024),
        prediction,
        prob_fake,
        featuresStr,
        componentsStr
      ]
    );

    return res.status(201).json({
      message: 'Prediction complete and saved.',
      id: saved.rows?.[0]?.id,
      ...result
    });
  } catch (error) {
    console.error('Prediction error:', error);
    return res.status(500).json({
      error: 'Failed to score and save review using the deep learning model'
    });
  }
});


// ---------- Continuous Logging Endpoint ----------
app.post('/api/predict/log', async (req, res) => {
  const { predictions } = req.body || {};
  if (!Array.isArray(predictions)) {
    return res.status(400).json({ error: 'predictions array required' });
  }

  let okCount = 0;
  for (const p of predictions) {
    try {
      const { text, rating, url, domain, prediction, prob_fake } = p || {};
      if (!text) continue;

      await pool.query(
        `INSERT INTO predict (
            review_text, rating, product_id, reviewer_id,
            prediction, prob_fake, features, components, created_at
         ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,NOW())`,
        [
          text,
          rating ?? null,
          domain ? String(domain).slice(0, 512) : null,
          url ? String(url).slice(0, 1024) : null,
          prediction,
          prob_fake,
          null,
          null
        ]
      );
      okCount++;
    } catch (e) {
      console.warn('predict/log insert failed for one item:', e.message);
    }
  }

  return res.json({ ok: true, count: okCount });
});


// ---------- Lightweight Health Check ----------
app.get('/api/health-ml', async (_req, res) => {
  try {
    const analytics = await runInference('analytics');
    res.json({ ok: true, analytics_sample: analytics });
  } catch (e) {
    res.status(500).json({ ok: false, error: String(e) });
  }
});


// --- API ENDPOINT (Updated for your schema) ---
app.post('/submit_review', async (req, res) => {
  // 🪵 Log incoming request body
  console.log("\n📩 Incoming Review Data:");
  console.log(req.body);

  const { userId, client_name, rating, reviewText } = req.body;

  // 🧠 Validation
  if (
    userId === undefined || userId === null ||
    !client_name ||
    rating === undefined || rating === null ||
    !reviewText
  ) {
    console.log("❌ Missing fields:", { userId, client_name, rating, reviewText });
    return res.status(400).json({ detail: 'All fields are required.' });
  }

  const queryText = `
    INSERT INTO reviews (user_id, client_name, rating, review_text)
    VALUES ($1, $2, $3, $4)
    RETURNING id
  `;
  const values = [userId, client_name, rating, reviewText];

  try {
    const result = await pool.query(queryText, values);
    console.log(" Review saved successfully:", result.rows[0]);
    res.status(201).json({ message: 'Review submitted successfully!' });
  } catch (err) {
    console.error(" Database error:", err.message);
    res.status(500).json({ detail: 'Failed to save review due to a server error.' });
  }
});

// Fetch saved analyses
app.get('/api/review-analysis', async (_req, res) => {
  try {
    const r = await pool.query('SELECT * FROM review_analysis ORDER BY created_at DESC LIMIT 500');
    res.json({ rows: r.rows, count: r.rowCount });
  } catch (e) {
    console.error('Error fetching review_analysis:', e);
    res.status(500).json({ error: 'Failed to fetch review_analysis' });
  }
});

app.listen(port, () => {
console.log(`Server is running on http://localhost:${port}`);
// Ensure schema on startup (idempotent)
ensureSchema();
// Warm-up model so first predict is fast
(async () => {
try {
console.log(' Warming up ML model...');
await runInference('predict', { text: 'warmup', rating: 5 });
console.log(' ML model warm-up complete.');
} catch (e) {
console.warn(' ML warm-up failed (continuing):', String(e?.message || e));
}
})();
});