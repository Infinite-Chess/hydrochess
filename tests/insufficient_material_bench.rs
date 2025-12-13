use hydrochess_wasm::board::{Board, Piece, PieceType, PlayerColor};
use hydrochess_wasm::evaluation::is_insufficient_material;
use std::time::Instant;

/// Create various test positions with different piece configurations
fn create_test_positions() -> Vec<Board> {
    let mut positions = Vec::new();
    
    // Position 1: King vs King (insufficient)
    let mut board1 = Board::new();
    board1.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board1.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board1);
    
    // Position 2: King + Bishop vs King (insufficient - same color bishops)
    let mut board2 = Board::new();
    board2.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board2.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White)); // light square
    board2.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board2);
    
    // Position 3: King + Knight vs King (insufficient)
    let mut board3 = Board::new();
    board3.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board3.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board3.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board3);
    
    // Position 4: King + Rook vs King (sufficient)
    let mut board4 = Board::new();
    board4.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board4.set_piece(0, 7, Piece::new(PieceType::Rook, PlayerColor::White));
    board4.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board4);
    
    // Position 5: King + Queen vs King (sufficient)
    let mut board5 = Board::new();
    board5.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board5.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board5.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board5);
    
    // Position 6: King + 2 Knights vs King (insufficient with standard rules)
    let mut board6 = Board::new();
    board6.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board6.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board6.set_piece(2, 1, Piece::new(PieceType::Knight, PlayerColor::White));
    board6.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board6);
    
    // Position 7: Lone Queen (0K) (insufficient)
    let mut board7 = Board::new();
    board7.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board7.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board7);
    
    // Position 8: 2 Rooks (0K) (insufficient)
    let mut board8 = Board::new();
    board8.set_piece(0, 0, Piece::new(PieceType::Rook, PlayerColor::White));
    board8.set_piece(1, 0, Piece::new(PieceType::Rook, PlayerColor::White));
    board8.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board8);
    
    // Position 9: King + Bishop + Knight vs King (insufficient)
    let mut board9 = Board::new();
    board9.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board9.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White));
    board9.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board9.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board9);
    
    // Position 10: King + 2 Bishops (opposite colors) vs King (insufficient)
    let mut board10 = Board::new();
    board10.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board10.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White)); // light
    board10.set_piece(2, 3, Piece::new(PieceType::Bishop, PlayerColor::White)); // dark
    board10.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    positions.push(board10);
    
    positions
}

#[test]
fn bench_insufficient_material_100k() {
    let positions = create_test_positions();
    let iterations = 100_000;
    
    println!("\n=== Insufficient Material Benchmark ===");
    println!("Testing {} positions, {} iterations each", positions.len(), iterations);
    
    let start = Instant::now();
    
    let mut results = vec![false; positions.len()];
    
    for _ in 0..iterations {
        for (i, board) in positions.iter().enumerate() {
            results[i] = is_insufficient_material(board);
        }
    }
    
    let elapsed = start.elapsed();
    let total_checks = iterations * positions.len();
    let per_check_ns = elapsed.as_nanos() / total_checks as u128;
    
    println!("\nResults:");
    println!("  Total time: {:?}", elapsed);
    println!("  Total checks: {}", total_checks);
    println!("  Time per check: {} ns", per_check_ns);
    println!("  Checks per second: {:.2}M", (total_checks as f64) / elapsed.as_secs_f64() / 1_000_000.0);
    
    // Print individual position results
    println!("\nPosition results:");
    for (i, result) in results.iter().enumerate() {
        let status = if *result { "DRAW (insufficient)" } else { "PLAYABLE (sufficient)" };
        println!("  Position {}: {}", i + 1, status);
    }
    
    // Basic sanity checks
    assert!(results[0], "K vs K should be insufficient");
    assert!(results[1], "K+B vs K should be insufficient");
    assert!(results[2], "K+N vs K should be insufficient");
    assert!(results[3], "K+R vs K should be sufficient");
    assert!(results[4], "K+Q vs K should be sufficient");
}

#[test]
fn test_insufficient_material_correctness() {
    // Test specific scenarios for correctness
    
    // K vs K
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert!(is_insufficient_material(&board), "K vs K should be insufficient");
    
    // K+B vs K (same color bishops only)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(2, 2, Piece::new(PieceType::Bishop, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert!(is_insufficient_material(&board), "K+B vs K should be insufficient");
    
    // K+N vs K
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(1, 2, Piece::new(PieceType::Knight, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert!(is_insufficient_material(&board), "K+N vs K should be insufficient");
    
    // K+R vs K (sufficient)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(0, 7, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert!(is_insufficient_material(&board), "K+R vs K should be insufficient");
    
    // K+Q vs K (sufficient)
    let mut board = Board::new();
    board.set_piece(0, 0, Piece::new(PieceType::King, PlayerColor::White));
    board.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    assert!(is_insufficient_material(&board), "K+Q vs K should be insufficient");
    
    // Lone Queen (0K) vs K - insufficient
    let mut board = Board::new();
    board.set_piece(3, 3, Piece::new(PieceType::Queen, PlayerColor::White));
    board.set_piece(7, 7, Piece::new(PieceType::King, PlayerColor::Black));
    assert!(is_insufficient_material(&board), "Lone Q vs K should be insufficient");
    
    // Lone Rook (0K) vs K - insufficient
    let mut board = Board::new();
    board.set_piece(0, 7, Piece::new(PieceType::Rook, PlayerColor::White));
    board.set_piece(4, 4, Piece::new(PieceType::King, PlayerColor::Black));
    assert!(is_insufficient_material(&board), "Lone R vs K should be insufficient");
    
    println!("All correctness tests passed!");
}
