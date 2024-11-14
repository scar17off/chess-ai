# Chess AI with Neural Network

A sophisticated chess engine powered by a neural network, featuring both a graphical user interface and a web interface for gameplay and training.

## Features

- Neural network-based chess AI
- Interactive chess board visualization
- Multiple game modes (AI vs Human, Free Play)
- Training system with grandmaster-style game generation
- Model management and persistence
- Support for common chess openings
- Real-time move evaluation visualization

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Generate training games:
```bash
python generate_training_games.py
```

## Usage

### Web Interface

Run the web interface with:
```bash
python webui.py
```

This launches a Gradio-based web interface with the following features:
- Interactive chess board
- Move input and validation
- AI move calculation
- Model selection and management
- Training game generation
- Model training interface

Features include:
- Pygame-based graphical interface
- Neural network visualization
- Real-time move evaluation
- Direct mouse interaction

## Training

The AI can be trained using:
1. Generated grandmaster-style games
2. Custom game recordings

To generate training games:
```bash
python generate_training_games.py
```

The system supports various grandmaster openings including:
- Sicilian Defense
- Ruy Lopez
- Italian Game
- French Defense
- Caro-Kann
- Queen's Gambit
- King's Indian
- And more...

## Configuration

Game parameters can be adjusted in `chess_config.json`:
```json
{
"DEFAULT_NUM_GAMES": 20,
"MIN_MOVES_PER_GAME": 30,
"MAX_MOVES_PER_GAME": 100,
"RANDOMIZATION_FACTOR": 0.2,
"CENTER_CONTROL_BONUS": 1.2,
"DEVELOPMENT_BONUS": 1.1,
"CHECK_BONUS": 0.5,
"CAPTURE_BONUS": 0.3
}
```

## Project Structure

- `webui.py`: Web interface implementation
- `neural_network.py`: Neural network model
- `game_recorder.py`: Game recording functionality
- `generate_training_games.py`: Training data generation
- `chess_config.json`: Configuration parameters
- `models/`: Saved model states
- `games/`: Recorded and generated games
- `pieces/`: Chess piece images

## Model Architecture

The neural network evaluates chess positions using:
- Piece positions and values
- Board control
- Development factors
- King safety
- Material balance
- Tactical opportunities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.