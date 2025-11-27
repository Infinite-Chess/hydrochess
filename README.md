# HydroChess WASM

A Rust-based chess engine compiled to WebAssembly for infinite chess variants.

## Features

### Engine Capabilities

- **Coordinate-based board**: Supports arbitrary piece positions (not limited to 8x8)
- **Multiple piece types**: Standard chess + fairy pieces (Amazon, Chancellor, Archbishop, Centaur, Hawk, Knightrider, etc.)
- **Iterative deepening search** with time management
- **Alpha-beta pruning** with aspiration windows
- **Null move pruning** and **late move reductions (LMR)**
- **Transposition table** with Zobrist hashing
- **Killer moves** and **history heuristic** for move ordering
- **Quiescence search** for tactical accuracy
- **Coordinate normalization** for infinite board positions

### WASM Interface

```rust
// Create engine from game state
let engine = Engine::new(json_state)?;

// Get best move (default time control)
let best_move = engine.get_best_move();

// Get best move with custom time limit
let best_move = engine.get_best_move_with_time(500); // 500ms

// Get all legal moves (for opening generation, UI, etc.)
let moves = engine.get_legal_moves_js();

// Run perft for testing
let nodes = engine.perft(5);
```

### Evaluation

- **Material counting** with piece values
- **Piece-square tables** for positional evaluation
- **King safety** evaluation
- **Pawn structure** analysis
- **Endgame detection** and specialized evaluation
- **Insufficient material** draw detection

## Building

### Browser Target (default)

```bash
wasm-pack build --target web
```

Output in `pkg/` - use with bundlers (esbuild, webpack, etc.)

### Node.js Target (for SPRT testing)

```bash
wasm-pack build --target nodejs --out-dir pkg-node
```

Output in `pkg-node/` - use with Node.js directly

## Usage in JavaScript

```javascript
import init, { Engine } from './pkg/hydrochess_wasm_v2.js';

await init();

// Game state in coordinate format
const gameState = {
    board: {
        pieces: [
            { x: "1", y: "1", piece_type: "r", player: "w" },
            { x: "5", y: "1", piece_type: "k", player: "w" },
            // ... more pieces
        ]
    },
    turn: "w",
    castling_rights: ["1,1", "5,1", "8,1"],  // Rook/King positions with rights
    en_passant: null,  // or { square: "x,y", pawn_square: "x,y" }
    halfmove_clock: 0,
    fullmove_number: 1,
    move_history: []
};

const engine = new Engine(gameState);
const bestMove = engine.get_best_move();
// Returns: { from: "5,2", to: "5,4", promotion: null }
```

## SPRT Testing

The engine includes a comprehensive SPRT (Sequential Probability Ratio Test) tool for comparing engine versions.

```bash
cd sprt

# First time setup
wasm-pack build --target nodejs --out-dir pkg-node

# Run SPRT test
node sprt.js run                    # Default settings
node sprt.js run gainer all         # Gainer bounds for weak engines
node sprt.js run nonreg top200      # Non-regression test
```

Features:
- **Automatic baseline/test management**: Snapshots old engine, rebuilds new
- **Parallel game playing**: Uses worker threads for speed
- **Standard SPRT bounds presets**: stockfish_ltc, stockfish_stc, top30, top200, all
- **Gainer vs non-regression modes**: Different hypothesis testing
- **Coordinate-based openings**: Uses engine's own legal move generation
- **Color-reversed pairs**: Each opening played with both color assignments

See [sprt/README.md](sprt/README.md) for full documentation.

## Project Structure

```
hydrochess-wasm/
├── src/
│   ├── lib.rs          # WASM bindings and Engine struct
│   ├── board.rs        # Board representation and piece types
│   ├── game.rs         # GameState and move making/unmaking
│   ├── moves.rs        # Move generation for all piece types
│   ├── search.rs       # Search algorithm (iterative deepening, alpha-beta)
│   ├── evaluation.rs   # Position evaluation
│   ├── normalization.rs # Coordinate normalization for infinite boards
│   └── utils.rs        # Utilities and panic hook
├── sprt/               # SPRT testing tool
│   ├── sprt.js         # Main SPRT script
│   ├── game-runner.js  # Game playing logic
│   └── wasm-loader.js  # Node.js WASM loader
├── pkg/                # Browser WASM build (generated)
├── pkg-node/           # Node.js WASM build (generated)
└── Cargo.toml          # Rust dependencies
```

## Piece Type Codes

| Code | Piece | Code | Piece |
|------|-------|------|-------|
| `p` | Pawn | `m` | Amazon |
| `n` | Knight | `c` | Chancellor |
| `b` | Bishop | `a` | Archbishop |
| `r` | Rook | `e` | Centaur |
| `q` | Queen | `d` | Royal Centaur |
| `k` | King | `h` | Hawk |
| `g` | Guard | `s` | Knightrider |
| `l` | Camel | `o` | Rose |
| `i` | Giraffe | `u` | Huygen |
| `z` | Zebra | `y` | Royal Queen |