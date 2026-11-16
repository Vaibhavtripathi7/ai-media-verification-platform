// @ts-check
import js from "@eslint/js";
import prettier from "eslint-config-prettier";
import globals from "globals";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["**/.venv/**"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  {
    files: ["apps/api/public/**/*.js"],
    languageOptions: { globals: globals.browser },
  },
  {
    files: ["apps/api/{src,test,scripts}/**/*.ts"],
    languageOptions: { globals: globals.node },
  },
  prettier,
);
