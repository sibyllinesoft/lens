// Utility functions
export function validateEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

export function generateId() {
  return Math.random().toString(36).substr(2, 9);
}

export class Logger {
  constructor(name) {
    this.name = name;
  }

  info(message) {
    console.log('[' + this.name + '] INFO: ' + message);
  }

  error(message) {
    console.error('[' + this.name + '] ERROR: ' + message);
  }
}