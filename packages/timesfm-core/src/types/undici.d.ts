/**
 * Type declarations for Node.js bundled `undici` module (Node ≥ 20).
 *
 * Node.js 20+ ships undici as the built-in HTTP client (exposed via fetch()).
 * The ProxyAgent is available via `import('undici')` at runtime for configuring
 * proxy support without environment variable mutation.
 *
 * This declaration file provides the minimal types needed by model-downloader.
 */

declare module 'undici' {
  export interface ProxyAgentOptions {
    uri: string;
    keepAliveTimeout?: number;
    keepAliveMaxTimeout?: number;
  }

  export class ProxyAgent {
    constructor(options: ProxyAgentOptions);
  }
}
