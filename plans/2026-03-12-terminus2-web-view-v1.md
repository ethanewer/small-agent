# Terminus 2 Agent Web View Implementation Plan

## Overview
Build a standalone JavaScript web view for running the Terminus 2 agent without modifying existing Python code. The web view will provide a modern UI for agent interaction with real-time updates.

## Architecture

### Directory Structure
```
web-view/
├── index.html          # Main HTML file with UI layout
├── styles.css          # CSS styling for the interface
├── app.js              # Frontend JavaScript for UI interactions
├── server.js           # Node.js backend server
├── package.json        # Node.js dependencies
└── README.md           # Setup and usage instructions
```

### Components

#### 1. Frontend (HTML/CSS/JS)
- **Instruction Input Area**: Text area for entering agent tasks
- **Configuration Panel**: Model and agent selection dropdowns
- **Output Display**: Real-time streaming of agent responses
  - Turn-by-turn reasoning (analysis/plan)
  - Command execution (input/output)
  - Terminal state updates
  - Error/warning messages
- **Status Indicators**: Running, completed, stopped states
- **Control Buttons**: Start, stop, clear

#### 2. Backend (Node.js Server)
- **WebSocket Server**: Real-time bidirectional communication
- **Python Agent Integration**: Spawn Python process running Terminus 2 agent
- **Event Stream**: Forward agent events to frontend
- **Configuration Handler**: Load and validate config.json

## Implementation Details

### Frontend Features
1. **Configuration Management**
   - Load available models from config.json
   - Load available agents from config.json
   - Set verbosity level (0 or 1)
   - Set max turns and max wait seconds

2. **Real-time Event Display**
   - `reasoning` events: Show analysis and plan in styled panels
   - `command_output` events: Display command keystrokes and output
   - `issue` events: Show errors and warnings
   - `done` events: Display completion message
   - `stopped` events: Show when max turns reached

3. **User Interactions**
   - Submit instruction for agent to process
   - Stop ongoing agent run
   - Clear output display
   - Adjust settings mid-session

### Backend Features
1. **Python Agent Integration**
   - Use `child_process` to spawn Python CLI
   - Pass instruction and configuration as arguments
   - Parse and forward agent events via WebSocket
   - Handle process lifecycle (start/stop)

2. **Event Streaming Protocol**
   - JSON-based event messages
   - Turn number tracking
   - Event type classification
   - Error handling and recovery

3. **Configuration Loading**
   - Read config.json from Python project
   - Resolve API keys from environment
   - Validate model and agent selections

## Technical Approach

### No Existing Code Modifications
- All changes in new `web-view/` directory
- Use existing Python CLI as black box
- Spawn Python process with appropriate arguments
- Parse output or use Python subprocess for event streaming

### Technology Stack
- **Frontend**: Vanilla HTML/CSS/JavaScript (no framework dependencies)
- **Backend**: Node.js with `ws` WebSocket library
- **Python Integration**: `child_process` module for subprocess management
- **Styling**: Modern CSS with flexbox/grid layout

## File Specifications

### index.html
- Responsive layout with sidebar for controls, main area for output
- Styled panels for reasoning, commands, errors
- Real-time update zones for each event type
- Clean, professional UI similar to CLI Rich panels

### styles.css
- Dark/light theme support
- Monospace fonts for terminal output
- Color-coded event types (cyan for reasoning, green for success, red for errors)
- Smooth animations for new content
- Responsive design for different screen sizes

### app.js
- WebSocket client for server communication
- Event handler for different event types
- DOM manipulation for real-time updates
- Form handling for configuration
- Error handling and user feedback

### server.js
- Express.js for HTTP (config serving)
- WebSocket server for real-time events
- Python subprocess management
- Event forwarding and message formatting
- Configuration endpoint for frontend

### package.json
- Dependencies: `ws`, `express`, `child_process` (built-in)
- Scripts for development and production

## User Flow

1. **Initial Load**: User opens web view in browser
2. **Configuration**: User selects model, agent, verbosity from dropdowns
3. **Instruction**: User enters task in text area
4. **Execution**: User clicks "Start Agent"
5. **Real-time Updates**: Agent events stream to output display
   - Reasoning panels appear with analysis/plan
   - Command outputs show keystrokes and results
   - Terminal state updates
6. **Completion**: Done message or stop signal
7. **Reset**: User can clear and start new task

## Error Handling
- Invalid configuration: Show error in UI
- Agent errors: Display in issue panel
- WebSocket disconnect: Show reconnection attempt
- Python process errors: Forward to frontend with details

## Future Enhancements (Not in v1)
- Session history persistence
- Multiple concurrent agent runs
- Export results to JSON
- Custom keyboard shortcuts
- Theme customization