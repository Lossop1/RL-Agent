import React, { Component, type ErrorInfo, type ReactNode } from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./styles.css";

class RootErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Locomotion console rendering failed", error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <main className="agent-layout loading-layout">
        <section className="primary-panel">
          <p className="eyebrow">运动策略控制台</p>
          <h1>页面暂时无法渲染</h1>
          <p>后端服务仍可继续运行，但前端初始化遇到异常。</p>
          <div className="inline-alert" role="alert">
            {this.state.error.message || "未知页面错误"}
          </div>
          <button className="secondary-button" onClick={() => window.location.reload()}>
            重新加载页面
          </button>
        </section>
      </main>
    );
  }
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <RootErrorBoundary>
      <App />
    </RootErrorBoundary>
  </React.StrictMode>
);
