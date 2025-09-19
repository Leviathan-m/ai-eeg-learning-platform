import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

// Async thunks for learning operations
export const getLearningContent = createAsyncThunk(
  'learning/getContent',
  async (params = {}, { rejectWithValue }) => {
    try {
      const response = await axios.get('/api/v1/learning/content', { params });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

export const startLearningSession = createAsyncThunk(
  'learning/startSession',
  async (sessionData, { rejectWithValue }) => {
    try {
      const response = await axios.post('/api/v1/learning/sessions', sessionData);
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

const learningSlice = createSlice({
  name: 'learning',
  initialState: {
    content: [],
    currentSession: null,
    progress: {},
    loading: false,
    error: null,
  },
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(getLearningContent.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getLearningContent.fulfilled, (state, action) => {
        state.loading = false;
        state.content = action.payload;
      })
      .addCase(getLearningContent.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export const { clearError } = learningSlice.actions;
export default learningSlice.reducer;
