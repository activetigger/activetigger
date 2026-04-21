module.exports = {
  root: true,
  env: { browser: true, es2020: true },
  extends: [
    'eslint:recommended',
    'plugin:@typescript-eslint/recommended',
    'plugin:react/recommended', // React recommended rules
    'plugin:react-hooks/recommended', // Recommended rules for React hooks
    'plugin:@typescript-eslint/eslint-recommended', // ESLint overrides for TypeScript
    'prettier', // Prettier rules
    'plugin:prettier/recommended', // Prettier plugin integration
  ],
  ignorePatterns: ['dist', '.eslintrc.cjs', 'generated'],
  parser: '@typescript-eslint/parser',
  plugins: ['@typescript-eslint', 'react-refresh'],
  rules: {
    'react/no-unescaped-entities': 'off',
    'react/prop-types': 'off',
    'react/react-in-jsx-scope': 'off',
    'react-refresh/only-export-components': ['warn', { allowConstantExport: true }],
    '@typescript-eslint/no-unused-vars': ['warn', { argsIgnorePattern: '^_' }],
    'react-hooks/exhaustive-deps': 'error',
  },
  settings: {
    react: {
      version: 'detect',
    },
  },
};
