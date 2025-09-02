// Example TypeScript file for search engine indexing
import { promises as fs } from 'fs';

export interface User {
  id: string;
  name: string;
  email: string;
  createdAt: Date;
}

export class UserService {
  private users: User[] = [];

  async createUser(userData: Omit<User, 'id' | 'createdAt'>): Promise<User> {
    const user: User = {
      id: Math.random().toString(36).substr(2, 9),
      ...userData,
      createdAt: new Date()
    };

    this.users.push(user);
    return user;
  }

  async findUserById(id: string): Promise<User | null> {
    const user = this.users.find(u => u.id === id);
    return user || null;
  }

  async findUsersByEmail(email: string): Promise<User[]> {
    return this.users.filter(u => u.email.includes(email));
  }

  async getAllUsers(): Promise<User[]> {
    return [...this.users];
  }

  async updateUser(id: string, updates: Partial<Omit<User, 'id'>>): Promise<User | null> {
    const userIndex = this.users.findIndex(u => u.id === id);
    if (userIndex === -1) return null;

    const currentUser = this.users[userIndex];
    if (!currentUser) return null;
    
    const updated: User = {
      id: currentUser.id,
      name: updates.name || currentUser.name,
      email: updates.email || currentUser.email,
      createdAt: updates.createdAt || currentUser.createdAt
    };
    this.users[userIndex] = updated;
    return updated;
  }

  async deleteUser(id: string): Promise<boolean> {
    const initialLength = this.users.length;
    this.users = this.users.filter(u => u.id !== id);
    return this.users.length < initialLength;
  }

  // Search functionality
  async searchUsers(query: string): Promise<User[]> {
    const lowercaseQuery = query.toLowerCase();
    return this.users.filter(user => 
      user.name.toLowerCase().includes(lowercaseQuery) ||
      user.email.toLowerCase().includes(lowercaseQuery)
    );
  }

  // Data persistence
  async saveToFile(filename: string): Promise<void> {
    const data = JSON.stringify(this.users, null, 2);
    await fs.writeFile(filename, data, 'utf-8');
  }

  async loadFromFile(filename: string): Promise<void> {
    try {
      const data = await fs.readFile(filename, 'utf-8');
      this.users = JSON.parse(data);
    } catch (error) {
      console.warn('Could not load users from file:', error);
      this.users = [];
    }
  }
}

// Utility functions
export function validateEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

export function generateUsername(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]/g, '')
    .substring(0, 15);
}

// Constants
export const MAX_USERS = 1000;
export const DEFAULT_USER_ROLE = 'user';
export const ADMIN_ROLE = 'admin';

// Example usage
async function main() {
  const userService = new UserService();
  
  // Create some sample users
  await userService.createUser({
    name: 'Alice Johnson',
    email: 'alice@example.com'
  });

  await userService.createUser({
    name: 'Bob Smith',
    email: 'bob@company.com'
  });

  await userService.createUser({
    name: 'Carol Davis',
    email: 'carol.davis@startup.io'
  });

  // Search for users
  const searchResults = await userService.searchUsers('alice');
  console.log('Search results:', searchResults);

  // Save data
  await userService.saveToFile('users.json');
}

if (require.main === module) {
  main().catch(console.error);
}