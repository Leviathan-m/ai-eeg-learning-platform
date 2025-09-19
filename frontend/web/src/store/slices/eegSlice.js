import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

// Async thunks for EEG operations
export const submitEEGData = createAsyncThunk(
  'eeg/submitData',
  async (eegData, { rejectWithValue }) => {
    try {
      const response = await axios.post('/api/v1/eeg/submit', eegData);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

export const getEEGSession = createAsyncThunk(
  'eeg/getSession',
  async (sessionId, { rejectWithValue }) => {
    try {
      const response = await axios.get(`/api/v1/eeg/sessions/${sessionId}`);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

export const getEEGHistory = createAsyncThunk(
  'eeg/getHistory',
  async (params = {}, { rejectWithValue }) => {
    try {
      const response = await axios.get('/api/v1/eeg/sessions', { params });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

export const getEEGStats = createAsyncThunk(
  'eeg/getStats',
  async (_, { rejectWithValue }) => {
    try {
      const response = await axios.get('/api/v1/eeg/stats');
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

const eegSlice = createSlice({
  name: 'eeg',
  initialState: {
    // Current session data
    currentSession: null,
    sessionId: null,
    isConnected: false,
    connectionStatus: 'disconnected',

    // Real-time data
    realTimeData: [],
    attentionScore: 0.5,
    stressLevel: 0.5,
    cognitiveLoad: 0.5,
    signalQuality: 0.0,

    // Historical data
    sessions: [],
    sessionDetails: null,

    // Statistics
    stats: null,

    // UI state
    loading: false,
    error: null,
    lastUpdate: null,

    // WebSocket connection
    wsConnection: null,
  },
  reducers: {
    // WebSocket connection management
    setConnectionStatus: (state, action) => {
      state.connectionStatus = action.payload;
      state.isConnected = action.payload === 'connected';
    },

    // Real-time data updates
    updateRealtimeData: (state, action) => {
      const { features, timestamp } = action.payload;

      state.realTimeData.push({
        ...features,
        timestamp
      });

      // Keep only last 100 data points
      if (state.realTimeData.length > 100) {
        state.realTimeData = state.realTimeData.slice(-100);
      }

      // Update current metrics
      state.attentionScore = features.attention_score || 0.5;
      state.stressLevel = features.stress_level || 0.5;
      state.cognitiveLoad = features.cognitive_load || 0.5;
      state.signalQuality = features.signal_quality || 0.0;
      state.lastUpdate = timestamp;
    },

    // Session management
    startSession: (state, action) => {
      state.currentSession = action.payload;
      state.sessionId = action.payload.session_id;
      state.realTimeData = [];
    },

    endSession: (state, action) => {
      state.currentSession = null;
      state.sessionId = null;
    },

    // Data management
    clearRealtimeData: (state) => {
      state.realTimeData = [];
    },

    // Error handling
    setError: (state, action) => {
      state.error = action.payload;
    },

    clearError: (state) => {
      state.error = null;
    },

    // WebSocket management
    setWSConnection: (state, action) => {
      state.wsConnection = action.payload;
    },

    disconnectWS: (state) => {
      if (state.wsConnection) {
        state.wsConnection.close();
        state.wsConnection = null;
      }
      state.connectionStatus = 'disconnected';
      state.isConnected = false;
    },
  },
  extraReducers: (builder) => {
    // Submit EEG data
    builder
      .addCase(submitEEGData.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(submitEEGData.fulfilled, (state, action) => {
        state.loading = false;
        if (!state.sessionId) {
          state.sessionId = action.payload.session_id;
        }
      })
      .addCase(submitEEGData.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })

      // Get EEG session
      .addCase(getEEGSession.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getEEGSession.fulfilled, (state, action) => {
        state.loading = false;
        state.sessionDetails = action.payload;
      })
      .addCase(getEEGSession.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })

      // Get EEG history
      .addCase(getEEGHistory.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getEEGHistory.fulfilled, (state, action) => {
        state.loading = false;
        state.sessions = action.payload;
      })
      .addCase(getEEGHistory.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      })

      // Get EEG stats
      .addCase(getEEGStats.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getEEGStats.fulfilled, (state, action) => {
        state.loading = false;
        state.stats = action.payload;
      })
      .addCase(getEEGStats.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

// Export actions
export const {
  setConnectionStatus,
  updateRealtimeData,
  startSession,
  endSession,
  clearRealtimeData,
  setError,
  clearError,
  setWSConnection,
  disconnectWS,
} = eegSlice.actions;

// Export selectors
export const selectEEGState = (state) => state.eeg;
export const selectRealtimeData = (state) => state.eeg.realTimeData;
export const selectCurrentMetrics = (state) => ({
  attentionScore: state.eeg.attentionScore,
  stressLevel: state.eeg.stressLevel,
  cognitiveLoad: state.eeg.cognitiveLoad,
  signalQuality: state.eeg.signalQuality,
});
export const selectConnectionStatus = (state) => ({
  isConnected: state.eeg.isConnected,
  status: state.eeg.connectionStatus,
});

export default eegSlice.reducer;
