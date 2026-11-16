# Build from the repository root: docker build -f apps/api/Dockerfile .
FROM node:24-alpine

RUN corepack enable
WORKDIR /srv

COPY package.json pnpm-lock.yaml pnpm-workspace.yaml ./
COPY apps/api/package.json apps/api/
RUN pnpm install --frozen-lockfile --prod --filter api

COPY apps/api/src apps/api/src
COPY apps/api/public apps/api/public

USER node
WORKDIR /srv/apps/api
ENV NODE_ENV=production
EXPOSE 3000
HEALTHCHECK --interval=10s --timeout=3s \
    CMD node -e "fetch('http://localhost:3000/health').then((r) => process.exit(r.ok ? 0 : 1), () => process.exit(1))"
CMD ["node", "src/server.ts"]
