/** @type {import("prettier").Config} */
//import prettierPluginSortImports from '@trivago/prettier-plugin-sort-imports';

const config = {
  printWidth: 100,
  singleQuote: true,
  arrowParens: 'always',
  proseWrap: 'always',
  trailingComma: 'all',
  importOrder: ['<THIRD_PARTY_MODULES>', '^[./]'],
  importOrderSeparation: true,
  importOrderSortSpecifiers: true,
  importOrderGroupNamespaceSpecifiers: true,
//  plugins: [prettierPluginSortImports],
};

export default config;
