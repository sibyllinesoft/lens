// User management service
export interface User {
  id: string;
  name: string;
  email: string;
}

export class UserService {
  private users: User[] = [];

  async createUser(userData: Omit<User, 'id'>): Promise<User> {
    const user: User = {
      id: Math.random().toString(36).substr(2, 9),
      ...userData
    };
    this.users.push(user);
    return user;
  }

  async findUserById(id: string): Promise<User | null> {
    return this.users.find(u => u.id === id) || null;
  }

  async searchUsers(query: string): Promise<User[]> {
    const q = query.toLowerCase();
    return this.users.filter(user => 
      user.name.toLowerCase().includes(q) ||
      user.email.toLowerCase().includes(q)
    );
  }
}