import { defineConfig } from '@vscode/test-cli';
import os from 'os';
import path from 'path';

export default defineConfig({
	files: 'test/**/*.test.js',
	// macOS limits unix socket paths to ~103 chars; keep user-data short.
	launchArgs: [
		'--user-data-dir',
		path.join(os.tmpdir(), 'vscode-llm-copilot-test'),
	],
});
