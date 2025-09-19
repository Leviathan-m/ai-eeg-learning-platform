-- AI-EEG Learning Platform Database Initialization
-- This script creates the initial database schema and inserts sample data

-- Create database if it doesn't exist
-- Note: This is handled by docker-compose environment variables

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Sample learning content data
INSERT INTO learning_content (
    content_id, title, subject, difficulty, description,
    content_type, duration_minutes, prerequisites, learning_objectives,
    tags, created_at
) VALUES
('intro_math_001', 'Introduction to Algebra', 'mathematics', 2,
 'Learn the fundamentals of algebraic expressions and equations',
 'video', 45, ARRAY['basic_arithmetic'], ARRAY['solve_linear_equations', 'understand_variables'],
 ARRAY['algebra', 'equations', 'variables'], NOW()),

('physics_mechanics_001', 'Newton''s Laws of Motion', 'physics', 3,
 'Explore the fundamental principles of classical mechanics',
 'interactive', 60, ARRAY['basic_math', 'vectors'], ARRAY['apply_newtons_laws', 'solve_motion_problems'],
 ARRAY['physics', 'mechanics', 'newton', 'motion'], NOW()),

('chemistry_basics_001', 'Atomic Structure and Periodic Table', 'chemistry', 2,
 'Understanding the building blocks of matter',
 'quiz', 30, ARRAY[], ARRAY['identify_elements', 'understand_atomic_structure'],
 ARRAY['chemistry', 'atoms', 'periodic_table', 'elements'], NOW()),

('biology_cells_001', 'Cell Biology Fundamentals', 'biology', 3,
 'Discover the basic unit of life and its functions',
 'video', 50, ARRAY['basic_science'], ARRAY['identify_cell_organelles', 'understand_cell_processes'],
 ARRAY['biology', 'cells', 'organelles', 'life_science'], NOW()),

('programming_python_001', 'Python Programming Basics', 'computer_science', 1,
 'Start your programming journey with Python',
 'interactive', 40, ARRAY[], ARRAY['write_basic_programs', 'understand_syntax'],
 ARRAY['programming', 'python', 'coding', 'beginner'], NOW()),

('english_grammar_001', 'English Grammar Essentials', 'english', 2,
 'Master the fundamental rules of English grammar',
 'text', 35, ARRAY[], ARRAY['identify_parts_of_speech', 'construct_sentences'],
 ARRAY['english', 'grammar', 'language', 'writing'], NOW()),

('history_world_war_2_001', 'World War II Overview', 'history', 3,
 'Understanding the causes, events, and consequences of WWII',
 'video', 55, ARRAY['basic_history'], ARRAY['identify_key_events', 'understand_global_impact'],
 ARRAY['history', 'world_war_2', '20th_century', 'global_conflict'], NOW()),

('art_drawing_001', 'Basic Drawing Techniques', 'art', 1,
 'Learn fundamental drawing skills and techniques',
 'interactive', 45, ARRAY[], ARRAY['use_basic_tools', 'apply_drawing_techniques'],
 ARRAY['art', 'drawing', 'creativity', 'visual_arts'], NOW()),

('music_theory_001', 'Music Theory Fundamentals', 'music', 2,
 'Understanding the language of music',
 'video', 40, ARRAY[], ARRAY['read_music_notation', 'understand_scales'],
 ARRAY['music', 'theory', 'notation', 'composition'], NOW()),

('economics_basics_001', 'Introduction to Economics', 'economics', 3,
 'Learn about supply, demand, and market dynamics',
 'text', 50, ARRAY['basic_math'], ARRAY['analyze_markets', 'understand_economic_principles'],
 ARRAY['economics', 'markets', 'supply_demand', 'finance'], NOW());

-- Sample user (for testing)
-- Password: testpassword (hashed with bcrypt)
-- In production, users should be created through the API
INSERT INTO users (
    username, email, hashed_password, full_name, is_active, created_at
) VALUES (
    'demo_user',
    'demo@example.com',
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/J8ZnEvlKO0tQqTcDK',
    'Demo User',
    true,
    NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_learning_content_subject_difficulty
ON learning_content(subject, difficulty);

CREATE INDEX IF NOT EXISTS idx_learning_content_created_at
ON learning_content(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_eeg_sessions_user_time
ON eeg_sessions(user_id, start_time DESC);

CREATE INDEX IF NOT EXISTS idx_learning_sessions_user_time
ON learning_sessions(user_id, start_time DESC);

CREATE INDEX IF NOT EXISTS idx_recommendations_user_time
ON recommendations(user_id, recommended_at DESC);

-- Comments for documentation
COMMENT ON TABLE users IS 'User accounts and profiles for the learning platform';
COMMENT ON TABLE eeg_sessions IS 'EEG recording sessions with device and quality information';
COMMENT ON TABLE eeg_data_points IS 'Individual EEG data samples with processed features';
COMMENT ON TABLE learning_content IS 'Educational content available in the platform';
COMMENT ON TABLE learning_sessions IS 'User learning activity sessions';
COMMENT ON TABLE recommendations IS 'AI-generated learning content recommendations';
COMMENT ON TABLE system_metrics IS 'System performance and monitoring metrics';
