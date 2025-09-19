# AI-EEG Learning Platform Frontend Dockerfile

FROM node:18-alpine AS builder

# Set environment variables
ENV NODE_ENV=production

# Install dependencies first for better caching
WORKDIR /app
COPY frontend/web/package*.json ./
RUN npm install --legacy-peer-deps --force && npm cache clean --force

# Copy source code
COPY frontend/web/ .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built application
COPY --from=builder /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY docker/nginx.conf /etc/nginx/conf.d/default.conf

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nextjs -u 1001

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000 || exit 1

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
