import tseslint from '@typescript-eslint/eslint-plugin';
import tsparser from '@typescript-eslint/parser';
import prettierConfig from 'eslint-config-prettier';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

// import.meta.dirname requires Node >= 21.2; use fallback for Node 20
const rootDir =
  typeof import.meta.dirname !== 'undefined'
    ? import.meta.dirname
    : dirname(fileURLToPath(import.meta.url));

export default [
  // Global ignores
  {
    ignores: ['**/dist/**', '**/node_modules/**', '**/coverage/**', '**/docs/api/**'],
  },

  // TypeScript source files (strict, full type-checking)
  {
    files: ['packages/*/src/**/*.ts'],
    languageOptions: {
      parser: tsparser,
      parserOptions: {
        projectService: true,
        tsconfigRootDir: rootDir,
      },
    },
    plugins: {
      '@typescript-eslint': tseslint,
    },
    rules: {
      // TypeScript strict rules
      '@typescript-eslint/no-explicit-any': 'error',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
      // no-non-null-assertion is redundant with tsconfig noUncheckedIndexedAccess:
      // TypeScript itself enforces undefined-checking on all indexed access,
      // so non-null assertions are explicit acknowledgements of verified bounds,
      // not a safety gap.
      '@typescript-eslint/no-non-null-assertion': 'off',
      '@typescript-eslint/consistent-type-imports': [
        'error',
        { prefer: 'type-imports', fixStyle: 'inline-type-imports' },
      ],

      // General best practices
      'no-console': ['warn', { allow: ['error', 'warn'] }],
      'no-debugger': 'error',
      'no-duplicate-imports': 'error',
      'prefer-const': 'error',
      'no-var': 'error',
    },
  },

  // Test files (no project references needed, relaxed rules)
  {
    files: ['packages/*/test/**/*.ts'],
    languageOptions: {
      parser: tsparser,
    },
    plugins: {
      '@typescript-eslint': tseslint,
    },
    rules: {
      '@typescript-eslint/no-explicit-any': 'off',
      'no-console': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
        },
      ],
    },
  },

  // JS scripts
  {
    files: ['scripts/**/*.js', '*.js'],
    rules: {
      'no-console': 'off',
    },
  },

  // Prettier — must be last to override formatting rules
  prettierConfig,
];
