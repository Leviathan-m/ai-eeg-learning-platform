import React from 'react';
import { Typography, Box, Card, CardContent, Button, Alert } from '@mui/material';
import { Psychology, WifiOff } from '@mui/icons-material';

const EEGMonitor = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        EEG Monitor
      </Typography>

      <Alert severity="info" sx={{ mb: 3 }}>
        Connect your EEG device to start real-time monitoring of your cognitive state.
      </Alert>

      <Card elevation={2}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Device Connection
          </Typography>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
            <WifiOff color="error" />
            <Typography variant="body1">
              No EEG device connected
            </Typography>
          </Box>

          <Button variant="contained" startIcon={<Psychology />}>
            Connect EEG Device
          </Button>

          <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
            Real-time EEG monitoring and cognitive state analysis will be displayed here.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default EEGMonitor;
