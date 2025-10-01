import React from 'react';
import { Typography, Box, Card, CardContent } from '@mui/material';

const Profile = () => {
  return (
    <Box>
      <Typography variant="h4" component="h1" gutterBottom>
        User Profile
      </Typography>

      <Card elevation={2}>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Profile Settings
          </Typography>
          <Typography variant="body2" color="text.secondary">
            User profile management and preferences will be available here.
          </Typography>
        </CardContent>
      </Card>
    </Box>
  );
};

export default Profile;
