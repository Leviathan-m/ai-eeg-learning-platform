import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

// Async thunks for analytics operations
export const getLearningAnalytics = createAsyncThunk(
  'analytics/getLearning',
  async (params = {}, { rejectWithValue }) => {
    try {
      const response = await axios.get('/api/v1/analytics/learning', { params });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

export const getEEGAnalytics = createAsyncThunk(
  'analytics/getEEG',
  async (params = {}, { rejectWithValue }) => {
    try {
      const response = await axios.get('/api/v1/analytics/eeg', { params });
      return response.data;
    } catch (error) {
      return rejectWithValue(error.response?.data || error.message);
    }
  }
);

const analyticsSlice = createSlice({
  name: 'analytics',
  initialState: {
    learning: null,
    eeg: null,
    dashboard: null,
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
      .addCase(getLearningAnalytics.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(getLearningAnalytics.fulfilled, (state, action) => {
        state.loading = false;
        state.learning = action.payload;
      })
      .addCase(getLearningAnalytics.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload;
      });
  },
});

export const { clearError } = analyticsSlice.actions;
export default analyticsSlice.reducer;
