import React, { useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  Button,
  Alert,
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  School,
  Timeline,
} from '@mui/icons-material';
import { useSelector, useDispatch } from 'react-redux';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

import { selectCurrentMetrics, selectConnectionStatus } from '../store/slices/eegSlice';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

const Dashboard = () => {
  const dispatch = useDispatch();
  const metrics = useSelector(selectCurrentMetrics);
  const connection = useSelector(selectConnectionStatus);

  // Sample data for charts
  const attentionData = {
    labels: ['10:00', '10:05', '10:10', '10:15', '10:20', '10:25', '10:30'],
    datasets: [
      {
        label: 'Attention Score',
        data: [0.6, 0.7, 0.8, 0.75, 0.9, 0.85, 0.88],
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        tension: 0.4,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'EEG Metrics Over Time',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
      },
    },
  };

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Learning Dashboard
      </Typography>

      {/* Connection Status Alert */}
      {!connection.isConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          EEG device not connected. Connect your EEG device to start monitoring your cognitive state.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Current Metrics Cards */}
        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Psychology color="primary" sx={{ mr: 1 }} />
                <Typography variant="h6">Attention</Typography>
              </Box>
              <Typography variant="h3" color="primary">
                {Math.round(metrics.attentionScore * 100)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.attentionScore * 100}
                sx={{ mt: 1, height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Current focus level
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <TrendingUp color="secondary" sx={{ mr: 1 }} />
                <Typography variant="h6">Stress Level</Typography>
              </Box>
              <Typography variant="h3" color="secondary">
                {Math.round(metrics.stressLevel * 100)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.stressLevel * 100}
                color="secondary"
                sx={{ mt: 1, height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                Cognitive load indicator
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <School color="success" sx={{ mr: 1 }} />
                <Typography variant="h6">Learning Progress</Typography>
              </Box>
              <Typography variant="h3" color="success">
                75%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={75}
                color="success"
                sx={{ mt: 1, height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                This week's progress
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card elevation={2}>
            <CardContent>
              <Box display="flex" alignItems="center" mb={2}>
                <Timeline color="info" sx={{ mr: 1 }} />
                <Typography variant="h6">Signal Quality</Typography>
              </Box>
              <Typography variant="h3" color="info">
                {Math.round(metrics.signalQuality * 100)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={metrics.signalQuality * 100}
                color="info"
                sx={{ mt: 1, height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                EEG signal strength
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* EEG Trends Chart */}
        <Grid item xs={12} md={8}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                EEG Metrics Trends
              </Typography>
              <Box sx={{ height: 300 }}>
                <Line data={attentionData} options={chartOptions} />
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Recommendations */}
        <Grid item xs={12} md={4}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                AI Recommendations
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography variant="body2" color="text.secondary" paragraph>
                  Based on your current attention level and learning patterns:
                </Typography>
                <Chip
                  label="Take a 5-minute break"
                  color="primary"
                  variant="outlined"
                  sx={{ mr: 1, mb: 1 }}
                />
                <Chip
                  label="Switch to visual content"
                  color="secondary"
                  variant="outlined"
                  sx={{ mr: 1, mb: 1 }}
                />
              </Box>
              <Button variant="contained" fullWidth>
                View All Recommendations
              </Button>
            </CardContent>
          </Card>
        </Grid>

        {/* Recent Learning Activity */}
        <Grid item xs={12}>
          <Card elevation={2}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Learning Activity
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                {[
                  {
                    title: 'Introduction to Algebra',
                    subject: 'Mathematics',
                    progress: 85,
                    lastAccessed: '2 hours ago',
                  },
                  {
                    title: "Newton's Laws of Motion",
                    subject: 'Physics',
                    progress: 60,
                    lastAccessed: '1 day ago',
                  },
                  {
                    title: 'Python Programming Basics',
                    subject: 'Computer Science',
                    progress: 100,
                    lastAccessed: '3 days ago',
                  },
                ].map((activity, index) => (
                  <Box
                    key={index}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      p: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1,
                    }}
                  >
                    <Box>
                      <Typography variant="subtitle1">{activity.title}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {activity.subject} • {activity.lastAccessed}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography variant="body2">
                        {activity.progress}% complete
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={activity.progress}
                        sx={{ width: 100, mt: 0.5 }}
                      />
                    </Box>
                  </Box>
                ))}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;
