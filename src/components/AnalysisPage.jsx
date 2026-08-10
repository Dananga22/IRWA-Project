import React, { useState } from 'react';
import { AGENT_DEFINITIONS, startDemoAnalysis } from '../services/analysisService';

export default function AnalysisPage() {
  const [demoState, setDemoState] = useState('idle'); // 'idle' | 'running' | 'completed'
  const [agentStatuses, setAgentStatuses] = useState({
    1: 'waiting',
    2: 'waiting',
    3: 'waiting',
    4: 'waiting',
    5: 'waiting'
  });

  const completedCount = Object.values(agentStatuses).filter(s => s === 'completed').length;
  const progressPercentage = Math.round((completedCount / 5) * 100);

  const handleStartDemo = () => {
    setDemoState('running');
    setAgentStatuses({ 1: 'waiting', 2: 'waiting', 3: 'waiting', 4: 'waiting', 5: 'waiting' });

    startDemoAnalysis(
      (agentId, status) => {
        setAgentStatuses(prev => ({ ...prev, [agentId]: status }));
      },
      () => {
        setDemoState('completed');
      }
    );
  };

  return (
    <div className="analysis-page-container">
      {/* Top Header Section */}
      <header className="analysis-header-card">
        <h1 className="main-title">Analyzing Your Medical Report</h1>
        <p className="subtitle">
          MedExplain AI is processing your report through a series of specialized AI agents.
        </p>

        <div className="report-meta-bar">
          <span>📄 <strong>Report:</strong> CBC_Laboratory_Report.pdf</span>
          <span>🌐 <strong>Language:</strong> English</span>
          <span>
            <strong>Status:</strong>{' '}
            <span className={`status-pill ${demoState === 'completed' ? 'complete' : 'active'}`}>
              {demoState === 'completed' ? '✓ Analysis Complete' : '⚙️ Processing...'}
            </span>
          </span>
        </div>

        <div className="safety-disclaimer-banner">
          ⚠️ <strong>Disclaimer:</strong> AI-generated analysis for educational purposes only. This is not a medical diagnosis.
        </div>
      </header>

      {/* Progress Bar Card */}
      <section className="overall-progress-card">
        <div className="progress-label-row">
          <span className="progress-title">Overall Progress</span>
          <span className="progress-text">
            <strong>{completedCount} of 5 agents completed</strong> ({progressPercentage}%)
          </span>
        </div>
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${progressPercentage}%` }}></div>
        </div>
      </section>

      {/* Main Agent Workflow Timeline */}
      <section className="workflow-timeline-section">
        <h2 className="section-heading">🤖 Multi-Agent Processing Workflow</h2>
        <div className="agent-cards-stack">
          {AGENT_DEFINITIONS.map(agent => {
            const status = agentStatuses[agent.id];
            return (
              <div key={agent.id} className={`agent-card-item status-${status}`}>
                <div className="card-top">
                  <div className="agent-identity">
                    <div className="agent-avatar-icon">{agent.icon}</div>
                    <div>
                      <span className="agent-tag">{agent.number} • {agent.technology}</span>
                      <h3 className="agent-heading">{agent.name}</h3>
                    </div>
                  </div>
                  <div className={`status-badge-chip badge-${status}`}>
                    {status === 'waiting' && '⏳ Waiting'}
                    {status === 'processing' && '⚙️ Processing...'}
                    {status === 'completed' && '✓ Completed'}
                  </div>
                </div>

                <p className="agent-description">{agent.description}</p>

                {/* Metadata & Architecture Badges */}
                <div className="badges-row">
                  {agent.badge === 'MCP TOOL CALL' && <span className="mcp-badge">⚡ MCP TOOL CALL</span>}
                  {agent.badge === 'Gemini API' && <span className="gemini-badge">✨ Gemini API</span>}
                  {agent.badge === 'Prototype — In Development' && (
                    <span className="prototype-badge">🛠️ Prototype — In Development</span>
                  )}
                  {agent.sources && (
                    <div className="sources-container">
                      <span className="sources-label">RAG Knowledge Base:</span>
                      {agent.sources.map(src => (
                        <span key={src} className="src-chip">{src}</span>
                      ))}
                    </div>
                  )}
                </div>

                {agent.mcpNote && <p className="mcp-tech-note">{agent.mcpNote}</p>}

                {/* Completion Details Box */}
                {status === 'completed' && (
                  <div className="agent-completion-box">
                    <p className="success-msg">{agent.outputSuccess}</p>
                    {agent.demoValues && (
                      <ul className="demo-values-preview">
                        {agent.demoValues.map((val, idx) => (
                          <li key={idx}>{val}</li>
                        ))}
                      </ul>
                    )}
                    {agent.checklist && (
                      <div className="checklist-grid">
                        {agent.checklist.map((item, idx) => (
                          <span key={idx} className="check-item">{item}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* LangGraph Shared State Diagram */}
      <section className="langgraph-state-panel">
        <h3>🔗 Agent Communication — LangGraph Shared State</h3>
        <div className="state-flow-diagram">
          <div className="state-node">Document Agent</div>
          <span className="flow-arrow">→ SharedState →</span>
          <div className="state-node">NLP Agent</div>
          <span className="flow-arrow">→ SharedState →</span>
          <div className="state-node">RAG Agent (MCP)</div>
          <span className="flow-arrow">→ SharedState →</span>
          <div className="state-node">LLM Agent</div>
          <span className="flow-arrow">→ SharedState →</span>
          <div className="state-node">Safety Agent</div>
        </div>
        <p className="langgraph-explanation">
          Agents exchange processing results through a shared state managed by LangGraph. The RAG capability is accessed through an MCP tool call.
        </p>
      </section>

      {/* Completion Banner */}
      {demoState === 'completed' && (
        <section className="completion-banner-card">
          <div className="party-icon">🎉</div>
          <h2>Analysis Complete ✓</h2>
          <p>Your report has been processed successfully.</p>
          <div className="button-row">
            <button className="view-explanation-btn" onClick={() => window.location.href = '/'}>
              View Explanation
            </button>
            <button className="return-dashboard-btn" onClick={() => window.location.href = '/'}>
              Return to Dashboard
            </button>
          </div>
        </section>
      )}

      {/* Floating Demo Control Bar */}
      <div className="demo-floating-bar">
        <span>⚡ Mid-Evaluation Live Demo</span>
        <button className="run-demo-btn" onClick={handleStartDemo} disabled={demoState === 'running'}>
          {demoState === 'running' ? 'Simulating Pipeline...' : '▶ Start Demo Analysis'}
        </button>
      </div>
    </div>
  );
}
