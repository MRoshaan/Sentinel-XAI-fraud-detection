import { Component, ReactNode } from "react";

export default class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <pre style={{ color: "red", background: "#1e293b", padding: 20, whiteSpace: "pre-wrap", fontSize: 13 }}>
          {this.state.error.message + "\n\n" + this.state.error.stack}
        </pre>
      );
    }
    return this.props.children;
  }
}
