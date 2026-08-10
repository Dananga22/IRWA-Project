/**
 * MedExplain AI — Analysis Service Layer
 * Clean abstraction separating UI components from backend communication.
 * In Phase 1 / Mid-Eval Demo, this service manages simulated demo progression.
 * In Phase 2+, replace simulateDemoAnalysis() with real POST /api/v1/upload endpoints.
 */

export const AGENT_DEFINITIONS = [
  {
    id: 1,
    number: "AGENT 1",
    name: "Document Processing Agent",
    icon: "📄",
    technology: "PyMuPDF",
    description: "Extracting text from the uploaded laboratory report using PyMuPDF.",
    duration: 2000,
    outputSuccess: "✓ Text extracted successfully (PDF → Raw Text)"
  },
  {
    id: 2,
    number: "AGENT 2",
    name: "NLP Extraction Agent",
    icon: "🔬",
    technology: "Regex / NLP",
    description: "Identifying laboratory test names, values, units, and reference ranges.",
    duration: 3000,
    outputSuccess: "✓ 8 laboratory values identified (Sample Demo Data):",
    demoValues: ["Hemoglobin — 13.2 g/dL", "WBC — 7.4 ×10⁹/L", "Platelets — 250 ×10⁹/L"]
  },
  {
    id: 3,
    number: "AGENT 3",
    name: "RAG Retrieval Agent",
    icon: "📚",
    technology: "Sentence Transformers + Vector DB",
    badge: "MCP TOOL CALL",
    description: "Retrieving relevant medical information from trusted sources.",
    duration: 3000,
    sources: ["WHO", "CDC", "MedlinePlus"],
    mcpNote: "Agent accesses the RAG retrieval capability through an MCP tool call.",
    outputSuccess: "✓ Relevant clinical sources retrieved via MCP Protocol"
  },
  {
    id: 4,
    number: "AGENT 4",
    name: "LLM Explanation Agent",
    icon: "🧠",
    technology: "Gemini API",
    badge: "Gemini API",
    description: "Generating a simple, patient-friendly explanation using the extracted results and retrieved medical context.",
    duration: 4000,
    outputSuccess: "✓ Patient-friendly explanation generated"
  },
  {
    id: 5,
    number: "AGENT 5",
    name: "Safety Verification Agent",
    icon: "🛡️",
    badge: "Prototype — In Development",
    description: "Reviewing the generated explanation for safety, clarity, and inappropriate medical claims.",
    duration: 3000,
    checklist: [
      "✓ No diagnosis generated",
      "✓ No unsafe treatment recommendation",
      "✓ Medical disclaimer included",
      "✓ Explanation reviewed"
    ],
    outputSuccess: "✓ Safety verification complete (Prototype Guardrail)"
  }
];

export const startDemoAnalysis = (onStepUpdate, onComplete) => {
  let currentIndex = 0;

  const runStep = () => {
    if (currentIndex >= AGENT_DEFINITIONS.length) {
      onComplete();
      return;
    }

    const currentAgent = AGENT_DEFINITIONS[currentIndex];
    onStepUpdate(currentAgent.id, 'processing');

    setTimeout(() => {
      onStepUpdate(currentAgent.id, 'completed');
      currentIndex += 1;
      runStep();
    }, currentAgent.duration);
  };

  runStep();
};
