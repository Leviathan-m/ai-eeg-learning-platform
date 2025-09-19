import React from 'react';
import { Typography, Box, Button, Card, CardContent } from '@mui/material';
import { Home } from '@mui/icons-material';
import { Link } from 'react-router-dom';

const NotFound = () => {
  return (
    <Box
      sx={{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '60vh',
      }}
    >
      <Card elevation={2} sx={{ maxWidth: 400, width: '100%' }}>
        <CardContent sx={{ textAlign: 'center', py: 4 }}>
          <Typography variant="h1" component="div" color="primary" sx={{ mb: 2 }}>
            404
          </Typography>

          <Typography variant="h5" component="h1" gutterBottom>
            Page Not Found
          </Typography>

          <Typography variant="body1" color="text.secondary" sx={{ mb: 3 }}>
            The page you're looking for doesn't exist or has been moved.
          </Typography>

          <Button
            variant="contained"
            startIcon={<Home />}
            component={Link}
            to="/"
            size="large"
          >
            Go Home
          </Button>
        </CardContent>
      </Card>
    </Box>
  );
};

export default NotFound;
