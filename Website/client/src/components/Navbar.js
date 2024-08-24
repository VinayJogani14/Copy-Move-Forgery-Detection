import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import AppBar from '@mui/material/AppBar';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import Menu from '@mui/material/Menu';
import MenuIcon from '@mui/icons-material/Menu';
import Container from '@mui/material/Container';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import { Stack } from '@mui/material';

import logo from './logo.JPG';

const pages = ['Home', 'Analytics'];

function ResponsiveAppBar() {
  const navigate = useNavigate();
  const location = useLocation();

  const [anchorElNav, setAnchorElNav] = React.useState(null);

  const handleOpenNavMenu = (event) => {
    setAnchorElNav(event.currentTarget);
  };

  const handleCloseNavMenu = () => {
    setAnchorElNav(false)
  };
  
  const handleNavigation = (page) => {
    if(page === "Home") navigate('/');
    else if (page === "get-started") navigate('/get-started');
    else if (page === "Analytics") navigate('/analytics');
  };

  return (
    <AppBar position="static" sx= {{ backgroundColor: 'white', boxShadow: 'none' }}>
      <Container maxWidth="lg" sx= {{ color: 'black' }}>
        <Toolbar disableGutters>
          <Stack sx={{ display: { xs: 'none', md: 'flex' } }} direction="row" spacing={1}>
            <img src={logo} width="10%" />
            <Typography
                    variant="h6"
                    noWrap
                    component="a"
                    href="/"
                    sx={{
                    ml: 1,
                    fontFamily: 'Montserrat',
                    fontWeight: 600,
                    letterSpacing: '.1rem',
                    color: 'inherit',
                    textDecoration: 'none',
                    }}
                >
                    TruthLens
            </Typography>
          </Stack>
          <Box sx={{ flexGrow: 1, display: { xs: 'flex', md: 'none' } }}>
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleOpenNavMenu}
              color="inherit"
            >
              <MenuIcon />
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorOrigin={{
                vertical: 'bottom',
                horizontal: 'left',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'left',
              }}
              open={Boolean(anchorElNav)}
              onClose={handleCloseNavMenu}
              sx={{
                display: { xs: 'block', md: 'none' },
              }}
            >
              {pages.map((page) => (
                <MenuItem key={page} onClick={handleCloseNavMenu}>
                  <Typography textAlign="center">{page}</Typography>
                </MenuItem>
              ))}
            </Menu>
          </Box>
          <Stack sx={{ display: { xs: 'flex', md: 'none' }, justifyContent: 'center' }} direction="row" spacing={1}>
            <img src={logo} width="15%" />
            <Typography
                    variant="h6"
                    noWrap
                    component="a"
                    href="/"
                    sx={{
                    ml: 1,
                    mb: 0,
                    fontFamily: 'Montserrat',
                    fontWeight: 600,
                    letterSpacing: '.1rem',
                    color: 'inherit',
                    textDecoration: 'none',
                    }}
                >
                    TruthLens
            </Typography>
          </Stack>
          <Box sx={{ flexGrow: 1, justifyContent: 'flex-end', display: { xs: 'none', md: 'flex' } }}>
            {pages.map((page) => (
              <Button
                key={page}
                onClick={() => handleNavigation(page)}
                sx={{ my: 2, color: '#212123', display: 'block', textTransform: 'none', fontFamily: 'Montserrat', fontSize: 16, mx: 1 }}
              >
                {page}
              </Button>
            ))}
          </Box>
          { location.pathname !== "/get-started" &&
          <Box sx={{ ml: 3 }}>
            <Button onClick={() => handleNavigation('get-started')} style={{ backgroundColor: '#14985C', color: 'white', textTransform: 'none', fontFamily: 'Montserrat' }} variant='contained'>
                Get Started
            </Button>
          </Box>
          }
        </Toolbar>
      </Container>
    </AppBar>
  );
}

export default ResponsiveAppBar;