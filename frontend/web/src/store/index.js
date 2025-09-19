import { configureStore } from '@reduxjs/toolkit';
import eegReducer from './slices/eegSlice';
import userReducer from './slices/userSlice';
import learningReducer from './slices/learningSlice';
import analyticsReducer from './slices/analyticsSlice';

export const store = configureStore({
  reducer: {
    eeg: eegReducer,
    user: userReducer,
    learning: learningReducer,
    analytics: analyticsReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        ignoredActions: ['persist/PERSIST', 'persist/REHYDRATE'],
      },
    }),
});

export default store;
