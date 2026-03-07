import { useState, useEffect, useRef } from "react";

const PRIMARY = "#0f0f1a";
const ACCENT = "#00e5ff";
const WARM = "#ff6b35";
const PURPLE = "#b388ff";
const GREEN = "#69ff47";
const CARD = "#161628";
const BORDER = "#2a2a4a";

const Section = ({ id, children, style = {} }) => (
  <section id={id} style={{ marginBottom: 64, ...style }}>{children}</section>
);

const SectionTitle = ({ emoji, children }) => (
  <div style={{ marginBottom: 28 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
      <span style={{ fontSize: 28 }}>{emoji}</span>
      <h2 style={{ fontSize: 26, fontFamily: "'Georgia', serif", color: "#fff", margin: 0, letterSpacing: "-0.5px" }}>{children}</h2>
    </div>
    <div style={{ height: 2, background: `linear-gradient(90deg, ${ACCENT}, transparent)`, width: 220 }} />
  </div>
);

const Card = ({ children, accent, style = {} }) => (
  <div style={{
    background: CARD,
    border: `1px solid ${accent || BORDER}`,
    borderRadius: 12,
    padding: "20px 24px",
    marginBottom: 16,
    boxShadow: accent ? `0 0 20px ${accent}18` : "none",
    ...style
  }}>
    {children}
  </div>
);

const Formula = ({ children }) => (
  <div style={{
    background: "#0a0a18",
    border: `1px solid ${BORDER}`,
    borderRadius: 8,
    padding: "14px 20px",
    fontFamily: "'Courier New', monospace",
    fontSize: 15,
    color: ACCENT,
    margin: "14px 0",
    overflowX: "auto",
    letterSpacing: 0.5
  }}>
    {children}
  </div>
);

const Tag = ({ children, color }) => (
  <span style={{
    background: color + "22",
    border: `1px solid ${color}`,
    borderRadius: 4,
    padding: "2px 8px",
    fontSize: 11,
    color,
    fontFamily: "monospace",
    fontWeight: 700,
    letterSpacing: 1
  }}>{children}</span>
);

// ─── DARTBOARD VISUAL ───────────────────────────────────────────────────
function Dartboard({ title, bias, variance, color, darts }) {
  const cx = 90, cy = 90, r = 70;
  const rings = [r, r * 0.72, r * 0.48, r * 0.26];
  return (
    <div style={{ textAlign: "center" }}>
      <svg width={180} height={180}>
        {rings.map((radius, i) => (
          <circle key={i} cx={cx} cy={cy} r={radius}
            fill={i % 2 === 0 ? "#1a1a2e" : "#11111f"}
            stroke={BORDER} strokeWidth={1} />
        ))}
        <circle cx={cx} cy={cy} r={8} fill="#e74c3c" />
        <circle cx={cx} cy={cy} r={3} fill="#fff" />
        {darts.map((d, i) => (
          <g key={i}>
            <circle cx={cx + d[0]} cy={cy + d[1]} r={5} fill={color} opacity={0.85} />
            <circle cx={cx + d[0]} cy={cy + d[1]} r={2} fill="#fff" opacity={0.9} />
          </g>
        ))}
      </svg>
      <div style={{ fontSize: 13, fontWeight: 700, color, marginBottom: 2 }}>{title}</div>
      <div style={{ fontSize: 11, color: "#888" }}>Bias: <span style={{ color: WARM }}>{bias}</span> · Var: <span style={{ color: PURPLE }}>{variance}</span></div>
    </div>
  );
}

// ─── ERROR CURVE ─────────────────────────────────────────────────────────
function ErrorCurve() {
  const w = 380, h = 200, padL = 48, padB = 36, padT = 20, padR = 20;
  const plotW = w - padL - padR, plotH = h - padB - padT;
  const xs = Array.from({ length: 80 }, (_, i) => i / 79);

  const bias2 = x => Math.exp(-3.5 * x) * 0.85 + 0.02;
  const variance = x => Math.pow(x, 2.2) * 0.9;
  const total = x => bias2(x) + variance(x) + 0.05;

  const toSVG = (x, y) => [padL + x * plotW, padT + (1 - Math.min(y, 1.1) / 1.1) * plotH];

  const path = (fn, color, dash = "") => {
    const pts = xs.map(x => toSVG(x, fn(x)));
    const d = "M" + pts.map(([px, py]) => `${px.toFixed(1)},${py.toFixed(1)}`).join("L");
    return <path d={d} fill="none" stroke={color} strokeWidth={2.5} strokeDasharray={dash} />;
  };

  const minX = xs.reduce((best, x) => total(x) < total(best) ? x : best, 0.5);
  const [mx, my] = toSVG(minX, total(minX));

  return (
    <svg width={w} height={h} style={{ display: "block", margin: "0 auto" }}>
      <rect width={w} height={h} fill="#0a0a18" rx={8} />
      {[0, 0.25, 0.5, 0.75, 1].map(v => {
        const [, py] = toSVG(0, v);
        return <line key={v} x1={padL} x2={w - padR} y1={py} y2={py} stroke={BORDER} strokeWidth={1} strokeDasharray="3,4" />;
      })}
      {path(bias2, WARM, "6,3")}
      {path(variance, PURPLE, "6,3")}
      {path(total, ACCENT)}
      <circle cx={mx} cy={my} r={6} fill={GREEN} />
      <text x={mx + 9} y={my - 5} fill={GREEN} fontSize={10} fontFamily="monospace">Sweet Spot</text>
      <text x={padL + plotW * 0.1} y={padT + plotH * 0.22} fill={WARM} fontSize={10}>Bias²</text>
      <text x={padL + plotW * 0.78} y={padT + plotH * 0.38} fill={PURPLE} fontSize={10}>Variance</text>
      <text x={padL + plotW * 0.45} y={padT + plotH * 0.1} fill={ACCENT} fontSize={10}>Total Error</text>
      <text x={padL} y={h - 6} fill="#555" fontSize={9}>Low Complexity</text>
      <text x={w - padR - 72} y={h - 6} fill="#555" fontSize={9}>High Complexity</text>
      <text x={6} y={padT + 4} fill="#555" fontSize={9} transform={`rotate(-90,6,${padT + 40})`}>Error</text>
      <line x1={padL} y1={padT} x2={padL} y2={h - padB} stroke="#444" strokeWidth={1} />
      <line x1={padL} y1={h - padB} x2={w - padR} y2={h - padB} stroke="#444" strokeWidth={1} />
    </svg>
  );
}

// ─── POLYNOMIAL FIT VISUAL ──────────────────────────────────────────────
function PolynomialFit() {
  const [degree, setDegree] = useState(1);
  const w = 340, h = 200;
  const pts = [
    [30, 145], [60, 120], [90, 100], [120, 90], [150, 105],
    [180, 88], [210, 75], [240, 85], [270, 70], [300, 60]
  ];

  const getLine = () => {
    if (degree === 1) {
      return `M30,148 L300,58`;
    } else if (degree === 3) {
      return `M30,148 C90,135 130,85 180,82 S260,68 300,62`;
    } else {
      return `M30,148 C50,80 70,160 90,102 S130,50 150,108 S200,45 240,88 S280,50 300,62`;
    }
  };

  const labels = { 1: ["High Bias", WARM, "Underfit"], 3: ["Good Fit", GREEN, "Just Right"], 8: ["High Variance", PURPLE, "Overfit"] };
  const [label, color, tag] = labels[degree];

  return (
    <div>
      <div style={{ display: "flex", gap: 10, marginBottom: 12, justifyContent: "center" }}>
        {[1, 3, 8].map(d => (
          <button key={d} onClick={() => setDegree(d)} style={{
            padding: "6px 16px", borderRadius: 6, border: `1px solid ${degree === d ? color : BORDER}`,
            background: degree === d ? color + "22" : "transparent", color: degree === d ? color : "#888",
            cursor: "pointer", fontSize: 12, fontFamily: "monospace", fontWeight: 700,
            transition: "all 0.2s"
          }}>
            Degree {d}
          </button>
        ))}
      </div>
      <svg width={w} height={h} style={{ display: "block", margin: "0 auto" }}>
        <rect width={w} height={h} fill="#0a0a18" rx={8} />
        {pts.map(([x, y], i) => (
          <circle key={i} cx={x} cy={y} r={5} fill="#fff" opacity={0.7} />
        ))}
        <path d={getLine()} fill="none" stroke={color} strokeWidth={2.5} strokeLinecap="round" />
        <text x={w / 2} y={h - 8} fill={color} fontSize={12} textAnchor="middle" fontFamily="monospace" fontWeight="bold">
          {label} — {tag}
        </text>
      </svg>
    </div>
  );
}

// ─── TRICK Q ────────────────────────────────────────────────────────────
function TrickQ({ q, a, warning }) {
  const [open, setOpen] = useState(false);
  return (
    <Card style={{ cursor: "pointer" }} accent={open ? ACCENT : null}>
      <div onClick={() => setOpen(!open)} style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
          <span style={{ fontSize: 18 }}>🎯</span>
          <p style={{ margin: 0, color: "#ccc", fontSize: 14, lineHeight: 1.5 }}>{q}</p>
        </div>
        <span style={{ color: ACCENT, fontSize: 18, marginLeft: 12 }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: `1px solid ${BORDER}` }}>
          {warning && (
            <div style={{ background: WARM + "15", border: `1px solid ${WARM}`, borderRadius: 6, padding: "8px 12px", marginBottom: 10, fontSize: 12, color: WARM }}>
              ⚠️ {warning}
            </div>
          )}
          <p style={{ margin: 0, color: "#bbb", fontSize: 13, lineHeight: 1.7 }}>{a}</p>
        </div>
      )}
    </Card>
  );
}

// ─── FAANG Q ────────────────────────────────────────────────────────────
function FaangQ({ q, level, a }) {
  const [open, setOpen] = useState(false);
  const lcolor = level === "HARD" ? WARM : PURPLE;
  return (
    <Card accent={open ? lcolor : null} style={{ cursor: "pointer" }}>
      <div onClick={() => setOpen(!open)} style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 12 }}>
        <div style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
          <Tag color={lcolor}>{level}</Tag>
          <p style={{ margin: 0, color: "#ddd", fontSize: 14, lineHeight: 1.5, marginTop: 2 }}>{q}</p>
        </div>
        <span style={{ color: lcolor, fontSize: 18 }}>{open ? "▲" : "▼"}</span>
      </div>
      {open && (
        <div style={{ marginTop: 14, paddingTop: 14, borderTop: `1px solid ${BORDER}` }}>
          {a}
        </div>
      )}
    </Card>
  );
}

// ─── MAIN ───────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState("concept");
  const tabs = [
    { id: "concept", label: "📐 Concept" },
    { id: "math", label: "🔢 Math" },
    { id: "visual", label: "🎨 Visuals" },
    { id: "tricks", label: "🎯 Tricks" },
    { id: "whatif", label: "🤔 What-Ifs" },
    { id: "faang", label: "🏢 FAANG" },
  ];

  return (
    <div style={{ background: PRIMARY, minHeight: "100vh", color: "#ddd", fontFamily: "'Georgia', serif", padding: "0 0 60px" }}>
      {/* HEADER */}
      <div style={{
        background: "linear-gradient(135deg, #0a0a18 0%, #101030 50%, #0a1a2a 100%)",
        borderBottom: `1px solid ${BORDER}`,
        padding: "40px 32px 32px",
        position: "relative",
        overflow: "hidden"
      }}>
        <div style={{
          position: "absolute", top: 0, left: 0, right: 0, bottom: 0,
          backgroundImage: `radial-gradient(circle at 20% 50%, ${ACCENT}08 0%, transparent 50%), radial-gradient(circle at 80% 20%, ${PURPLE}08 0%, transparent 50%)`
        }} />
        <div style={{ position: "relative" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 10 }}>
            <Tag color={ACCENT}>BIAS-VARIANCE</Tag>
            <Tag color={PURPLE}>DEEP DIVE</Tag>
          </div>
          <h1 style={{ fontSize: 36, fontFamily: "'Georgia', serif", margin: "0 0 10px", color: "#fff", letterSpacing: "-1px", lineHeight: 1.1 }}>
            The Bias-Variance<br />
            <span style={{ color: ACCENT }}>Tradeoff</span>
          </h1>
          <p style={{ margin: 0, color: "#888", fontSize: 14, lineHeight: 1.6, maxWidth: 520 }}>
            The most fundamental tension in machine learning — why every model is a compromise between being wrong in one of two completely different ways.
          </p>
        </div>
      </div>

      {/* TABS */}
      <div style={{
        display: "flex", gap: 4, padding: "16px 32px", borderBottom: `1px solid ${BORDER}`,
        background: "#0d0d1e", overflowX: "auto", flexWrap: "wrap"
      }}>
        {tabs.map(t => (
          <button key={t.id} onClick={() => setActiveTab(t.id)} style={{
            padding: "8px 16px", borderRadius: 6,
            border: `1px solid ${activeTab === t.id ? ACCENT : "transparent"}`,
            background: activeTab === t.id ? ACCENT + "15" : "transparent",
            color: activeTab === t.id ? ACCENT : "#666",
            cursor: "pointer", fontSize: 13, fontFamily: "monospace", fontWeight: 700,
            transition: "all 0.2s", whiteSpace: "nowrap"
          }}>
            {t.label}
          </button>
        ))}
      </div>

      {/* CONTENT */}
      <div style={{ maxWidth: 760, margin: "0 auto", padding: "40px 24px" }}>

        {/* ── CONCEPT ── */}
        {activeTab === "concept" && (
          <div>
            <SectionTitle emoji="💡">The Core Idea</SectionTitle>
            <Card>
              <p style={{ margin: "0 0 12px", lineHeight: 1.8, fontSize: 15 }}>
                <strong style={{ color: ACCENT }}>Machine learning is function approximation.</strong> Given data, you're trying to learn the true underlying function — but you can never see it. You only see noisy samples.
              </p>
              <p style={{ margin: 0, lineHeight: 1.8, fontSize: 14, color: "#aaa" }}>
                Every model fails in exactly one of two ways: it's either <em style={{ color: WARM }}>too rigid</em> to capture real patterns (high bias), or <em style={{ color: PURPLE }}>too flexible</em> and memorizes noise (high variance). You cannot be both at once.
              </p>
            </Card>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 24 }}>
              <Card accent={WARM}>
                <div style={{ fontSize: 22, marginBottom: 8 }}>📏</div>
                <div style={{ fontWeight: 700, color: WARM, marginBottom: 6, fontSize: 14 }}>HIGH BIAS</div>
                <p style={{ margin: 0, fontSize: 13, color: "#aaa", lineHeight: 1.7 }}>
                  Model makes strong, wrong assumptions. Like fitting a straight line to a curve — it systematically misses patterns. Same mistake every time.
                </p>
                <div style={{ marginTop: 10, fontSize: 12, color: "#666" }}>→ Underfitting · Training error high</div>
              </Card>
              <Card accent={PURPLE}>
                <div style={{ fontSize: 22, marginBottom: 8 }}>🌀</div>
                <div style={{ fontWeight: 700, color: PURPLE, marginBottom: 6, fontSize: 14 }}>HIGH VARIANCE</div>
                <p style={{ margin: 0, fontSize: 13, color: "#aaa", lineHeight: 1.7 }}>
                  Model is so sensitive it memorizes every quirk of training data — including noise. Change the data slightly and the model changes dramatically.
                </p>
                <div style={{ marginTop: 10, fontSize: 12, color: "#666" }}>→ Overfitting · Train/test gap large</div>
              </Card>
            </div>

            <SectionTitle emoji="🎯">The Dartboard Analogy</SectionTitle>
            <p style={{ fontSize: 14, color: "#999", marginBottom: 20, lineHeight: 1.7 }}>
              Think of training your model 100 times on different samples. Each "dart" is one model's prediction on the same test point. The bullseye is the true answer.
            </p>
            <div style={{
              display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 16,
              background: CARD, padding: 20, borderRadius: 12, border: `1px solid ${BORDER}`
            }}>
              <Dartboard title="Low Bias Low Var" bias="✅" variance="✅" color={GREEN}
                darts={[[5, -3], [-4, 6], [3, 5], [-5, -4], [4, -6], [-3, 3]]} />
              <Dartboard title="High Bias Low Var" bias="❌" variance="✅" color={WARM}
                darts={[[28, 30], [32, 26], [26, 32], [30, 28], [34, 30], [28, 34]]} />
              <Dartboard title="Low Bias High Var" bias="✅" variance="❌" color={PURPLE}
                darts={[[5, -35], [-38, 20], [30, 15], [-20, -30], [40, -10], [-15, 38]]} />
              <Dartboard title="High Bias High Var" bias="❌" variance="❌" color="#e74c3c"
                darts={[[35, 28], [-25, 40], [40, -30], [28, -38], [-38, 20], [20, 35]]} />
            </div>

            <SectionTitle emoji="📖" style={{ marginTop: 36 }}>Real World Examples</SectionTitle>

            <Card>
              <div style={{ fontWeight: 700, color: WARM, marginBottom: 8, fontSize: 13, fontFamily: "monospace" }}>HIGH BIAS EXAMPLES</div>
              <ul style={{ margin: 0, paddingLeft: 20, color: "#aaa", fontSize: 13, lineHeight: 2 }}>
                <li>Predicting house prices using only square footage (ignoring location, age, amenities)</li>
                <li>Using linear regression when the true relationship is quadratic</li>
                <li>A doctor always diagnosing based only on age, ignoring all symptoms</li>
                <li>KNN with K = N (predict the global average always)</li>
              </ul>
            </Card>
            <Card>
              <div style={{ fontWeight: 700, color: PURPLE, marginBottom: 8, fontSize: 13, fontFamily: "monospace" }}>HIGH VARIANCE EXAMPLES</div>
              <ul style={{ margin: 0, paddingLeft: 20, color: "#aaa", fontSize: 13, lineHeight: 2 }}>
                <li>A decision tree with no depth limit memorizing training labels perfectly</li>
                <li>GPT fine-tuned on 10 examples — memorizes those 10, fails elsewhere</li>
                <li>KNN with K = 1 (ultra-sensitive to nearest neighbor, even if noisy)</li>
                <li>A polynomial of degree 15 fitted to 20 data points</li>
              </ul>
            </Card>

            <SectionTitle emoji="⚖️">The Irreducible Error</SectionTitle>
            <Card accent={ACCENT}>
              <p style={{ margin: 0, fontSize: 14, lineHeight: 1.8 }}>
                Even a perfect model has error. This is called <strong style={{ color: ACCENT }}>irreducible error</strong> or <em>noise</em> — the inherent randomness in the data itself. A patient's blood pressure varies hour to hour regardless of the model. A stock price depends on unknowable future events. No model can remove this. It sets a hard floor on performance.
              </p>
            </Card>
          </div>
        )}

        {/* ── MATH ── */}
        {activeTab === "math" && (
          <div>
            <SectionTitle emoji="🔢">The Decomposition</SectionTitle>
            <p style={{ fontSize: 14, color: "#999", lineHeight: 1.7, marginBottom: 16 }}>
              The expected test error of any model trained on dataset D, evaluated at point x, decomposes exactly into three terms. This is a theorem — not an approximation.
            </p>

            <Card accent={ACCENT}>
              <div style={{ fontSize: 13, color: "#888", marginBottom: 8, fontFamily: "monospace" }}>THE MASTER EQUATION</div>
              <Formula>
                E[(y - ŷ)²] = Bias² + Variance + σ²(noise)
              </Formula>
              <Formula>
                E[(y - ŷ)²] = (E[ŷ] - f(x))² + E[(ŷ - E[ŷ])²] + σ²
              </Formula>
              <p style={{ margin: 0, fontSize: 12, color: "#666" }}>
                Where expectation is over all possible training datasets of fixed size, and f(x) is the true function.
              </p>
            </Card>

            <SectionTitle emoji="🔬">Term by Term</SectionTitle>

            <Card accent={WARM}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: WARM, fontSize: 14 }}>① Bias²</div>
                <Tag color={WARM}>Systematic Error</Tag>
              </div>
              <Formula>(E[ŷ] - f(x))²</Formula>
              <p style={{ margin: 0, fontSize: 13, color: "#aaa", lineHeight: 1.7 }}>
                <strong style={{ color: "#ddd" }}>E[ŷ]</strong> is your model's average prediction across many training sets. <strong style={{ color: "#ddd" }}>f(x)</strong> is the true answer. If these differ, your model has a systematic blind spot — it's wrong in a predictable direction, every single time, regardless of how much data you give it.
              </p>
            </Card>

            <Card accent={PURPLE}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: PURPLE, fontSize: 14 }}>② Variance</div>
                <Tag color={PURPLE}>Sensitivity Error</Tag>
              </div>
              <Formula>E[(ŷ - E[ŷ])²]</Formula>
              <p style={{ margin: 0, fontSize: 13, color: "#aaa", lineHeight: 1.7 }}>
                How much does your model's prediction fluctuate from one training set to another? High variance means your model's answer depends heavily on which specific samples you happened to collect. It's unstable.
              </p>
            </Card>

            <Card accent={"#555"}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div style={{ fontWeight: 700, color: "#888", fontSize: 14 }}>③ Irreducible Noise (σ²)</div>
                <Tag color="#555">Cannot Fix</Tag>
              </div>
              <Formula>σ² = Var(ε) where y = f(x) + ε</Formula>
              <p style={{ margin: 0, fontSize: 13, color: "#aaa", lineHeight: 1.7 }}>
                The variance of the true noise process. Even if you knew the exact true function f(x), y would still vary around it randomly. This is the theoretical minimum error — your ceiling.
              </p>
            </Card>

            <SectionTitle emoji="📐">The Full Derivation</SectionTitle>
            <Card>
              <p style={{ fontSize: 13, color: "#999", marginBottom: 12, lineHeight: 1.7 }}>
                Start with expected squared error, add and subtract E[ŷ] (a classic algebraic trick):
              </p>
              <Formula>
                {`E[(y - ŷ)²]`}
                <br />{`= E[(y - E[ŷ] + E[ŷ] - ŷ)²]`}
                <br />{`= E[(y - E[ŷ])²] + 2·E[(y-E[ŷ])(E[ŷ]-ŷ)] + E[(E[ŷ]-ŷ)²]`}
              </Formula>
              <p style={{ fontSize: 13, color: "#999", marginBottom: 8, lineHeight: 1.7 }}>
                The cross-term vanishes (ŷ is independent of y given x, and E[E[ŷ]-ŷ]=0), leaving:
              </p>
              <Formula>
                {`= E[(y - f(x))²]   (noise σ²)`}
                <br />{`+ (E[ŷ] - f(x))²  (Bias²)`}
                <br />{`+ E[(ŷ - E[ŷ])²]  (Variance)`}
              </Formula>
              <div style={{ fontSize: 12, color: "#555", fontStyle: "italic" }}>
                This decomposition holds exactly for MSE. For other losses, similar but different decompositions exist.
              </div>
            </Card>

            <SectionTitle emoji="📊">KNN as a Concrete Example</SectionTitle>
            <Card>
              <p style={{ fontSize: 14, color: "#aaa", marginBottom: 12, lineHeight: 1.7 }}>
                K-Nearest Neighbors is the perfect toy model because K directly controls the bias-variance tradeoff:
              </p>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
                <div style={{ background: "#0a0a18", borderRadius: 8, padding: 14 }}>
                  <div style={{ color: WARM, fontFamily: "monospace", fontSize: 12, fontWeight: 700, marginBottom: 6 }}>K = 1 (High Variance)</div>
                  <Formula>{`ŷ = y(nearest neighbor)`}</Formula>
                  <p style={{ margin: 0, fontSize: 12, color: "#888" }}>One noisy point determines everything. Wiggly, unstable decision boundary.</p>
                </div>
                <div style={{ background: "#0a0a18", borderRadius: 8, padding: 14 }}>
                  <div style={{ color: ACCENT, fontFamily: "monospace", fontSize: 12, fontWeight: 700, marginBottom: 6 }}>K = N (High Bias)</div>
                  <Formula>{`ŷ = mean(all y)`}</Formula>
                  <p style={{ margin: 0, fontSize: 12, color: "#888" }}>Predicts global average for everything. Zero variance, maximum bias.</p>
                </div>
              </div>
              <div style={{ marginTop: 12, padding: 12, background: GREEN + "10", borderRadius: 8, border: `1px solid ${GREEN}40`, fontSize: 13, color: "#bbb", lineHeight: 1.7 }}>
                As K increases: Variance ↓ monotonically, Bias ↑ monotonically. Total error has a minimum at some K* — that's the sweet spot.
              </div>
            </Card>
          </div>
        )}

        {/* ── VISUAL ── */}
        {activeTab === "visual" && (
          <div>
            <SectionTitle emoji="📈">Error vs. Model Complexity</SectionTitle>
            <p style={{ fontSize: 14, color: "#999", lineHeight: 1.7, marginBottom: 16 }}>
              The classic curve. As complexity grows, bias drops fast but variance climbs. Their sum — total error — forms a U-shape with a sweet spot somewhere in the middle.
            </p>
            <Card>
              <ErrorCurve />
              <div style={{ display: "flex", gap: 20, justifyContent: "center", marginTop: 14, flexWrap: "wrap" }}>
                <div style={{ fontSize: 12, color: WARM }}>━ ━ Bias²</div>
                <div style={{ fontSize: 12, color: PURPLE }}>━ ━ Variance</div>
                <div style={{ fontSize: 12, color: ACCENT }}>——— Total Error</div>
                <div style={{ fontSize: 12, color: GREEN }}>● Sweet Spot</div>
              </div>
            </Card>

            <SectionTitle emoji="🎨">Polynomial Fit — See It Live</SectionTitle>
            <p style={{ fontSize: 14, color: "#999", lineHeight: 1.7, marginBottom: 16 }}>
              Click the degree buttons to see underfitting vs. overfitting on the same dataset.
            </p>
            <Card>
              <PolynomialFit />
              <div style={{ marginTop: 14, display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 10 }}>
                <div style={{ fontSize: 12, color: "#888", textAlign: "center" }}>
                  <div style={{ color: WARM, fontWeight: 700 }}>Degree 1</div>
                  Misses the curve. High bias.
                </div>
                <div style={{ fontSize: 12, color: "#888", textAlign: "center" }}>
                  <div style={{ color: GREEN, fontWeight: 700 }}>Degree 3</div>
                  Captures the trend. Balanced.
                </div>
                <div style={{ fontSize: 12, color: "#888", textAlign: "center" }}>
                  <div style={{ color: PURPLE, fontWeight: 700 }}>Degree 8</div>
                  Follows noise. High variance.
                </div>
              </div>
            </Card>

            <SectionTitle emoji="🖼️">Reference Images</SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
              <Card>
                <div style={{ fontSize: 11, color: "#555", marginBottom: 8, fontFamily: "monospace" }}>CLASSIC BIAS-VARIANCE CURVE</div>
                <img
                  src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Bias_and_variance_contributing_to_total_error.svg/500px-Bias_and_variance_contributing_to_total_error.svg.png"
                  alt="Bias Variance Error Curve"
                  style={{ width: "100%", borderRadius: 8, opacity: 0.9 }}
                />
              </Card>
              <Card>
                <div style={{ fontSize: 11, color: "#555", marginBottom: 8, fontFamily: "monospace" }}>L1 vs L2 CONSTRAINT REGIONS</div>
                <img
                  src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Lasso_and_ridge_regression.png/400px-Lasso_and_ridge_regression.png"
                  alt="L1 vs L2 Regularization"
                  style={{ width: "100%", borderRadius: 8, opacity: 0.9 }}
                />
                <div style={{ fontSize: 11, color: "#555", marginTop: 6 }}>Regularization moves us left on the complexity curve</div>
              </Card>
            </div>

            <SectionTitle emoji="🗺️">The Landscape Map</SectionTitle>
            <div style={{ overflowX: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13, fontFamily: "monospace" }}>
                <thead>
                  <tr style={{ background: "#0a0a18" }}>
                    {["", "Low Variance", "High Variance"].map(h => (
                      <th key={h} style={{ padding: "10px 14px", textAlign: "left", color: "#888", borderBottom: `1px solid ${BORDER}`, fontWeight: 700 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ padding: "12px 14px", color: WARM, fontWeight: 700, borderBottom: `1px solid ${BORDER}20` }}>Low Bias</td>
                    <td style={{ padding: "12px 14px", color: GREEN, borderBottom: `1px solid ${BORDER}20` }}>✅ IDEAL — generalizes well</td>
                    <td style={{ padding: "12px 14px", color: PURPLE, borderBottom: `1px solid ${BORDER}20` }}>Overfits — needs regularization or more data</td>
                  </tr>
                  <tr>
                    <td style={{ padding: "12px 14px", color: WARM, fontWeight: 700 }}>High Bias</td>
                    <td style={{ padding: "12px 14px", color: WARM }}>Underfits — wrong model family</td>
                    <td style={{ padding: "12px 14px", color: "#e74c3c" }}>💀 Worst case — complex AND wrong</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <SectionTitle emoji="🏋️">Train vs. Test Error Signatures</SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              {[
                { scenario: "High Bias", train: "HIGH", test: "HIGH", note: "Both errors high. Model is too simple.", color: WARM },
                { scenario: "High Variance", train: "LOW", test: "HIGH", note: "Big gap. Model memorizes training data.", color: PURPLE },
                { scenario: "Good Fit", train: "LOW", test: "LOW", note: "Small gap. Generalizes well.", color: GREEN },
                { scenario: "Noisy Data", train: "HIGH", test: "SIMILAR", note: "Both high but small gap. Irreducible noise.", color: "#888" },
              ].map(({ scenario, train, test, note, color }) => (
                <Card key={scenario} accent={color}>
                  <div style={{ fontWeight: 700, color, marginBottom: 8, fontSize: 13 }}>{scenario}</div>
                  <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
                    <span style={{ fontSize: 11, background: "#111", padding: "2px 8px", borderRadius: 4, color: "#aaa" }}>Train: <strong style={{ color }}>{train}</strong></span>
                    <span style={{ fontSize: 11, background: "#111", padding: "2px 8px", borderRadius: 4, color: "#aaa" }}>Test: <strong style={{ color }}>{test}</strong></span>
                  </div>
                  <p style={{ margin: 0, fontSize: 12, color: "#888" }}>{note}</p>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* ── TRICKS ── */}
        {activeTab === "tricks" && (
          <div>
            <SectionTitle emoji="🎯">Trick Questions</SectionTitle>
            <p style={{ fontSize: 14, color: "#999", lineHeight: 1.7, marginBottom: 20 }}>
              These are the questions that trip up even experienced engineers. Click each one to reveal the answer.
            </p>

            <TrickQ
              q="More data always helps. True or false?"
              warning="This is the #1 misconception in ML interviews."
              a="FALSE — or at least, incomplete. More data reduces VARIANCE (model is less sensitive to which specific samples it saw). But it does NOT reduce BIAS. If your model family is fundamentally wrong (e.g., linear on a quadratic relationship), infinite data still won't help — you'll converge to the best linear fit, which is still wrong."
            />
            <TrickQ
              q="Your training loss is 0. Is your model perfect?"
              a="Absolutely not. Zero training loss means the model memorized the training data, possibly including all the noise. This is extreme overfitting — high variance. The only metric that matters is how it performs on UNSEEN data. A model that memorizes 10,000 training examples is useless if it fails on the 10,001st."
            />
            <TrickQ
              q="A more complex model always has higher variance. True?"
              warning="Wikipedia itself mentions this is a fallacy!"
              a="Not necessarily! Ensemble methods like Random Forest are complex (many trees) but have LOWER variance than a single deep tree, because averaging decorrelates the noise. The bias-variance tradeoff is about the model FAMILY and training procedure, not just raw complexity."
            />
            <TrickQ
              q="You add more features to your model. What happens to bias and variance?"
              a="Generally: more features → lower bias (richer hypothesis space, can fit more complex patterns) + higher variance (more degrees of freedom, more sensitive to specific data). BUT: if a feature is pure noise (uninformative), it only adds variance with zero reduction in bias. Feature selection is important!"
            />
            <TrickQ
              q="High bias and high variance simultaneously — is that possible?"
              a="Yes! A classic example: a neural network trained on the wrong task, or a model that's complex in the wrong dimensions. For instance, a high-degree polynomial that's been poorly regularized can oscillate wildly (high variance) AND still miss the global trend (high bias). It's the worst of both worlds."
            />
            <TrickQ
              q="Can validation loss be LOWER than training loss? What would that mean?"
              a="Yes, this happens! Most common cause: dropout is active during training (randomly zeros neurons → higher effective training loss) but disabled at test time → lower test loss. It can also happen with early stopping on small datasets or if train/val split was unlucky. It does NOT mean the model is 'better' at validation — it's a measurement artifact."
            />
            <TrickQ
              q="Does regularization always reduce variance?"
              warning="Think carefully about what regularization actually does."
              a="Yes, regularization reduces variance (it constrains the model, making it less sensitive to the specific training data). But it does so by INCREASING bias (it pushes the model away from the data-fit solution toward something simpler/constrained). λ controls the tradeoff: higher λ → less variance, more bias."
            />
            <TrickQ
              q="If two models have the same bias, should you always pick the one with lower variance?"
              a="Usually yes, but not always. Consider: (1) Computational cost — a lower-variance model might be 1000x more expensive to train. (2) Calibration — a slightly higher-variance model might be better calibrated for uncertainty estimates. (3) If you can ensemble the high-variance models, you can reduce variance while maintaining low bias."
            />
          </div>
        )}

        {/* ── WHAT-IFS ── */}
        {activeTab === "whatif" && (
          <div>
            <SectionTitle emoji="🤔">What-If Scenarios</SectionTitle>
            <p style={{ fontSize: 14, color: "#999", lineHeight: 1.7, marginBottom: 20 }}>
              These scenarios test whether you can apply the bias-variance lens to real situations.
            </p>

            {[
              {
                q: "What if you have only 50 training samples for a complex task?",
                answer: `You're in the high-variance danger zone. With few samples, even a moderately complex model will overfit. Your move:
• Use simple models (logistic regression, shallow tree, KNN with high K)
• Strong L2 regularization to constrain parameter space
• Cross-validation to reliably estimate true performance
• Transfer learning — borrow a pretrained model's low-bias representations
• Bayesian methods that quantify and naturally handle uncertainty
• Data augmentation if the domain permits`,
                color: PURPLE
              },
              {
                q: "What if training loss is decreasing but validation loss is increasing?",
                answer: `Textbook overfitting. Your model has memorized the training set but isn't generalizing. The variance term is dominating. Actions in order of aggressiveness:
1. Early stopping — just stop here (simplest)
2. Add dropout or L2 regularization
3. Reduce model capacity (fewer layers/neurons)
4. Get more training data
5. Data augmentation
The key insight: you're at a point where the model is past the sweet spot on the error curve — more training is making it worse.`,
                color: WARM
              },
              {
                q: "What if both train and test loss are high and plateaued?",
                answer: `This is HIGH BIAS — underfitting. The model isn't capturing the real signal. More data won't help (it reduces variance, not bias). Your moves:
• Increase model capacity (more layers, wider, higher-degree polynomial)
• Remove regularization (you're already underfitting — penalizing less)
• Add more informative features (feature engineering)
• Try a different model family altogether
• Check if the label is actually learnable from your features`,
                color: ACCENT
              },
              {
                q: "What if you add 10,000 more training examples — does bias change?",
                answer: `No. Bias is a property of the MODEL FAMILY and the algorithm, not the amount of data. If you're fitting a line to a quadratic relationship, more data gives you the BEST possible line — but it's still a line. The systematic error from wrong assumptions doesn't shrink with data. What DOES shrink: variance. The model becomes more stable, less sensitive to the specific training set you saw.`,
                color: GREEN
              },
              {
                q: "What if you average the predictions of 100 independent models?",
                answer: `Averaging (ensembling) is a direct attack on variance. For models with independent errors:
Var(average) = Var(single model) / 100
Bias(average) = Bias(single model)  ← unchanged!
This is exactly why bagging (Random Forest) works: it reduces variance without touching bias. If your models have correlated errors (trained on the same data), the variance reduction is less dramatic. The key is decorrelation — which is why Random Forest uses random feature subsets.`,
                color: PURPLE
              },
              {
                q: "What if your features have high noise — what happens to bias and variance?",
                answer: `Noisy features increase variance (the model latches onto noise differently each training run) and can also increase effective bias if the signal-to-noise ratio is so low that the model can't learn the true pattern at all. Solutions: feature selection (remove noisy features), regularization (shrink their coefficients), PCA/dimensionality reduction (filter noise directions), or collect cleaner data.`,
                color: WARM
              },
            ].map(({ q, answer, color }) => (
              <Card key={q} accent={color} style={{ marginBottom: 16 }}>
                <div style={{ display: "flex", gap: 10, marginBottom: 12 }}>
                  <span style={{ fontSize: 18 }}>🤔</span>
                  <p style={{ margin: 0, color: "#ddd", fontSize: 14, lineHeight: 1.5, fontWeight: 600 }}>{q}</p>
                </div>
                <div style={{ background: "#0a0a18", borderRadius: 8, padding: "14px 16px", borderLeft: `3px solid ${color}` }}>
                  <p style={{ margin: 0, fontSize: 13, color: "#bbb", lineHeight: 1.9, whiteSpace: "pre-line" }}>{answer}</p>
                </div>
              </Card>
            ))}
          </div>
        )}

        {/* ── FAANG ── */}
        {activeTab === "faang" && (
          <div>
            <SectionTitle emoji="🏢">FAANG Q&A</SectionTitle>
            <p style={{ fontSize: 14, color: "#999", lineHeight: 1.7, marginBottom: 20 }}>
              Real-style interview questions on bias-variance from Google, Meta, Amazon, Apple, Netflix engineering interviews. Click to expand the model answer.
            </p>

            <div style={{ marginBottom: 10, fontSize: 12, color: "#555", fontFamily: "monospace" }}>── MEDIUM ──────────────────────────</div>

            <FaangQ level="MEDIUM" q="How would you diagnose whether your model is suffering from high bias or high variance without seeing the test set?"
              a={
                <div>
                  <p style={{ fontSize: 13, color: "#bbb", lineHeight: 1.7, marginBottom: 12 }}>Use <strong style={{ color: ACCENT }}>learning curves</strong> — plot training and validation error as a function of training set size.</p>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                    <div style={{ background: "#0a0a18", borderRadius: 8, padding: 12 }}>
                      <div style={{ color: WARM, fontWeight: 700, fontSize: 12, marginBottom: 6 }}>HIGH BIAS SIGNATURE</div>
                      <ul style={{ margin: 0, paddingLeft: 16, color: "#999", fontSize: 12, lineHeight: 1.8 }}>
                        <li>Train error is HIGH</li>
                        <li>Val error ≈ Train error</li>
                        <li>Small gap between them</li>
                        <li>More data doesn't help</li>
                      </ul>
                    </div>
                    <div style={{ background: "#0a0a18", borderRadius: 8, padding: 12 }}>
                      <div style={{ color: PURPLE, fontWeight: 700, fontSize: 12, marginBottom: 6 }}>HIGH VARIANCE SIGNATURE</div>
                      <ul style={{ margin: 0, paddingLeft: 16, color: "#999", fontSize: 12, lineHeight: 1.8 }}>
                        <li>Train error is LOW</li>
                        <li>Val error is HIGH</li>
                        <li>Large gap between them</li>
                        <li>More data gradually helps</li>
                      </ul>
                    </div>
                  </div>
                  <p style={{ margin: "12px 0 0", fontSize: 12, color: "#777" }}>Bonus: K-fold cross-validation variance across folds is a direct measure of variance.</p>
                </div>
              }
            />

            <FaangQ level="MEDIUM" q="You're building a model for a client. They have very little data (200 samples) but want the best possible accuracy. Walk me through your strategy."
              a={
                <div style={{ fontSize: 13, color: "#bbb", lineHeight: 1.9 }}>
                  <p style={{ margin: "0 0 10px" }}>With 200 samples, you're in high-variance territory. Every decision should prioritize variance reduction:</p>
                  <p style={{ margin: "0 0 6px", color: ACCENT, fontWeight: 700, fontSize: 12 }}>1. Establish a baseline first</p>
                  <p style={{ margin: "0 0 10px" }}>Simple models (logistic regression, linear SVM) — understand the floor before adding complexity.</p>
                  <p style={{ margin: "0 0 6px", color: ACCENT, fontWeight: 700, fontSize: 12 }}>2. Use all data efficiently</p>
                  <p style={{ margin: "0 0 10px" }}>K-fold cross-validation (k=5 or 10) — don't waste any data on a held-out test set until the very end.</p>
                  <p style={{ margin: "0 0 6px", color: ACCENT, fontWeight: 700, fontSize: 12 }}>3. Transfer learning</p>
                  <p style={{ margin: "0 0 10px" }}>If applicable, start from a pretrained model. The pretrained weights already encode low-bias representations — fine-tuning requires far less data.</p>
                  <p style={{ margin: "0 0 6px", color: ACCENT, fontWeight: 700, fontSize: 12 }}>4. Regularize aggressively</p>
                  <p style={{ margin: "0 0 10px" }}>L2 regularization, early stopping, dropout — all directly reduce variance.</p>
                  <p style={{ margin: 0, color: "#555", fontSize: 12 }}>Trade-off to articulate: I'm accepting slightly higher bias (from regularization/simpler model) to dramatically reduce variance. With 200 samples, variance is the enemy.</p>
                </div>
              }
            />

            <FaangQ level="MEDIUM" q="Explain why Random Forest reduces variance but Gradient Boosting reduces bias. What does this tell you about when to use each?"
              a={
                <div style={{ fontSize: 13, color: "#bbb", lineHeight: 1.9 }}>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
                    <div style={{ background: "#0a0a18", borderRadius: 8, padding: 12 }}>
                      <div style={{ color: PURPLE, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>RANDOM FOREST → ↓ Variance</div>
                      <p style={{ margin: 0, fontSize: 12, color: "#999", lineHeight: 1.7 }}>Trains many high-variance (deep) trees independently on bootstrap samples with random feature subsets. Averaging decorrelated models: Var(avg) ≈ Var(tree)/n. Bias stays the same as a single tree.</p>
                    </div>
                    <div style={{ background: "#0a0a18", borderRadius: 8, padding: 12 }}>
                      <div style={{ color: WARM, fontWeight: 700, fontSize: 12, marginBottom: 8 }}>GRADIENT BOOSTING → ↓ Bias</div>
                      <p style={{ margin: 0, fontSize: 12, color: "#999", lineHeight: 1.7 }}>Sequentially adds shallow trees, each correcting the RESIDUALS of the previous. Each step reduces systematic error. Can overfit (increase variance) with too many rounds.</p>
                    </div>
                  </div>
                  <p style={{ margin: 0 }}><strong style={{ color: "#ddd" }}>When to use which:</strong> If your baseline model has good concepts but is unstable → RF. If your baseline systematically misses patterns → GBM. In practice: GBM usually wins on tabular data (at FAANG scale), RF is more robust with less tuning.</p>
                </div>
              }
            />

            <div style={{ marginBottom: 10, marginTop: 24, fontSize: 12, color: "#555", fontFamily: "monospace" }}>── HARD ─────────────────────────────</div>

            <FaangQ level="HARD" q="Explain double descent. Does it invalidate the bias-variance tradeoff?"
              a={
                <div style={{ fontSize: 13, color: "#bbb", lineHeight: 1.9 }}>
                  <p style={{ margin: "0 0 10px" }}>Classical theory predicts a U-shaped test error curve. But in 2019, Belkin et al. showed that beyond the <em>interpolation threshold</em> (where the model perfectly fits training data), test error can actually decrease again — a second descent.</p>
                  <Formula>
                    {"Under-parameterized → [U-curve] → Interpolation Threshold → Over-parameterized → [another descent]"}
                  </Formula>
                  <p style={{ margin: "10px 0" }}>At the interpolation threshold, test error spikes because there's exactly one solution that fits training data — and it's unstable. Beyond it, there are INFINITELY MANY zero-training-loss solutions. Gradient descent implicitly finds the <strong style={{ color: ACCENT }}>minimum norm solution</strong> among these — which has provable generalization properties.</p>
                  <p style={{ margin: "0 0 10px" }}>This is why GPT-3 (175B parameters, massively overparameterized) still generalizes.</p>
                  <p style={{ margin: 0, color: "#777", fontSize: 12 }}>Does it invalidate the tradeoff? No — it extends it. The B-V decomposition still holds at every point; the surprise is that the minimum is not where classical theory expected it. Implicit regularization from gradient descent shifts the landscape.</p>
                </div>
              }
            />

            <FaangQ level="HARD" q="At Google scale, you have 10B training samples. Is bias-variance still relevant? What changes?"
              a={
                <div style={{ fontSize: 13, color: "#bbb", lineHeight: 1.9 }}>
                  <p style={{ margin: "0 0 10px" }}>Absolutely still relevant, but the dominant concern shifts dramatically:</p>
                  <p style={{ margin: "0 0 6px", color: WARM, fontWeight: 700, fontSize: 12 }}>Variance is essentially solved</p>
                  <p style={{ margin: "0 0 10px" }}>With 10B samples, variance terms shrink to near-zero for any reasonable model. Your model's predictions are extremely stable across different samples of this size.</p>
                  <p style={{ margin: "0 0 6px", color: WARM, fontWeight: 700, fontSize: 12 }}>Bias becomes the main enemy</p>
                  <p style={{ margin: "0 0 10px" }}>The residual error is almost entirely bias — wrong model family, wrong features, wrong inductive biases. This is why Google pours effort into architecture research (Transformers, MoE), not data collection (past a point).</p>
                  <p style={{ margin: "0 0 6px", color: WARM, fontWeight: 700, fontSize: 12 }}>New challenges emerge</p>
                  <ul style={{ margin: "0 0 10px", paddingLeft: 20, color: "#999", lineHeight: 1.8 }}>
                    <li>Distribution mismatch between training and serving</li>
                    <li>Label noise at scale has large aggregate effect</li>
                    <li>Optimization at scale (learning rate scheduling, batch size)</li>
                    <li>Memorization of rare training examples becomes a privacy/security concern</li>
                  </ul>
                  <p style={{ margin: 0, color: "#777", fontSize: 12 }}>Interviewer follow-up: "What if data is noisy at that scale?" — The irreducible error term grows. More data doesn't help if all of it is noisy. Quality > quantity past a threshold.</p>
                </div>
              }
            />

            <FaangQ level="HARD" q="A junior engineer proposes to reduce variance by averaging 1,000 models trained on the same dataset. Will it work? What's the ceiling?"
              a={
                <div style={{ fontSize: 13, color: "#bbb", lineHeight: 1.9 }}>
                  <p style={{ margin: "0 0 10px" }}>Partially — and there's a fundamental ceiling. For models trained on the same fixed dataset:</p>
                  <Formula>{"Var(avg of N models) = ρ·σ² + (1-ρ)·σ²/N"}</Formula>
                  <p style={{ margin: "10px 0" }}>Where ρ is the pairwise correlation between model errors and σ² is individual model variance. As N→∞:</p>
                  <Formula>{"Var → ρ·σ²"}</Formula>
                  <p style={{ margin: "10px 0" }}>If models are trained identically (same data, same architecture, just different random seeds), their errors are highly correlated (ρ ≈ 0.8-0.9). You get maybe 10-20% variance reduction, then it plateaus.</p>
                  <p style={{ margin: "0 0 10px" }}><strong style={{ color: ACCENT }}>To actually reduce variance you need decorrelation:</strong></p>
                  <ul style={{ margin: "0 0 10px", paddingLeft: 20, color: "#999", lineHeight: 1.8 }}>
                    <li>Different training subsets (bagging)</li>
                    <li>Different feature subsets (random subspace method)</li>
                    <li>Different architectures (true ensemble)</li>
                    <li>Different hyperparameters</li>
                  </ul>
                  <p style={{ margin: 0, color: "#777", fontSize: 12 }}>The bias is completely unchanged — this proposal does nothing for bias. If the model is high-bias, you'd be averaging 1,000 copies of the same wrong answer.</p>
                </div>
              }
            />
          </div>
        )}

      </div>

      {/* FOOTER */}
      <div style={{ textAlign: "center", padding: "20px", color: "#333", fontSize: 11, fontFamily: "monospace", borderTop: `1px solid ${BORDER}` }}>
        BIAS-VARIANCE DEEP DIVE · ML INTERVIEW MASTERY
      </div>
    </div>
  );
}
