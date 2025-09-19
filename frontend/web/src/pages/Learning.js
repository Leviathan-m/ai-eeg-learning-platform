import React from 'react';
import { Typography, Box, Card, CardContent, Button, Grid, Chip } from '@mui/material';
import { School, PlayArrow } from '@mui/icons-material';

const Learning = () => {
  const sampleContent = [
    {
      id: 1,
      title: 'Introduction to Algebra',
      subject: 'Mathematics',
      difficulty: 2,
      duration: '45 min',
      description: 'Learn the fundamentals of algebraic expressions and equations',
    },
    {
      id: 2,
      title: "Newton's Laws of Motion",
      subject: 'Physics',
      difficulty: 3,
      duration: '60 min',
      description: 'Explore the fundamental principles of classical mechanics',
    },
    {
      id: 3,
      title: 'Python Programming Basics',
      subject: 'Computer Science',
      difficulty: 1,
      duration: '40 min',
      description: 'Start your programming journey with Python',
    },
  ];

  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        Learning Content
      </Typography>

      <Grid container spacing={3}>
        {sampleContent.map((content) => (
          <Grid item xs={12} md={6} lg={4} key={content.id}>
            <Card elevation={2}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {content.title}
                </Typography>

                <Box sx={{ mb: 2 }}>
                  <Chip
                    label={content.subject}
                    size="small"
                    color="primary"
                    sx={{ mr: 1 }}
                  />
                  <Chip
                    label={`Level ${content.difficulty}`}
                    size="small"
                    color="secondary"
                    sx={{ mr: 1 }}
                  />
                  <Chip
                    label={content.duration}
                    size="small"
                    variant="outlined"
                  />
                </Box>

                <Typography variant="body2" color="text.secondary" paragraph>
                  {content.description}
                </Typography>

                <Button
                  variant="contained"
                  startIcon={<PlayArrow />}
                  fullWidth
                >
                  Start Learning
                </Button>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </Box>
  );
};

export default Learning;
