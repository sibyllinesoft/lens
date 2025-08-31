# Suggested Commands for Lens Development

## Development Workflow
```bash
# Start development server with hot reload
npm run dev

# Build the project
npm run build

# Start production server
npm start
```

## Testing
```bash
# Run all tests
npm test

# Run tests with coverage report
npm run test:coverage

# Run tests with UI
npm run test:ui
```

## Code Quality
```bash
# Lint TypeScript code
npm run lint

# Format code with prettier
npm run fmt
```

## Architecture Validation
```bash
# Validate CUE architecture specification
npm run validate:config

# Check production configuration compliance
npm run validate:production

# Direct CUE commands
cue eval architecture.cue
cue export architecture.cue --expression lens_production
```

## Docker
```bash
# Build and start services with Docker Compose
docker-compose up --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## System Utilities (Linux)
- `ls -la` - List files with details
- `find . -name "*.ts" -type f` - Find TypeScript files
- `grep -r "pattern" src/` - Search in source files
- `git status` - Check git status
- `git log --oneline` - View commit history