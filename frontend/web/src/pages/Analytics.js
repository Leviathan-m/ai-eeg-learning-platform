import React from 'react';
import { Typography, Box, Card, CardContent, Grid } from '@mui/material';

const Analytics = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Analytics Dashboard
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Learning Progress
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Comprehensive analytics of your learning journey will be displayed here.
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={6}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                EEG Insights
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Cognitive performance analytics based on EEG data.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Analytics;
