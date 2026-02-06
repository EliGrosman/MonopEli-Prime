export type PropertyColor =
  | 'brown'
  | 'lightblue'
  | 'magenta'
  | 'orange'
  | 'red'
  | 'yellow'
  | 'green'
  | 'blue'
  | 'railroad'
  | 'utility';

export type SpaceType =
  | 'property'
  | 'railroad'
  | 'utility'
  | 'tax'
  | 'chance'
  | 'community_chest'
  | 'corner';

export interface PropertyState {
  position: number;
  owner: number | null;
  houses: number; // 0-4 for houses, 5 for hotel
  mortgaged: boolean;
}

export interface PropertyInfo {
  position: number;
  name: string;
  type: SpaceType;
  color?: PropertyColor;
  price: number;
  rent: number[];
  buildCost: number;
  mortgageValue: number;
}

export interface BoardSpace {
  position: number;
  name: string;
  type: SpaceType;
  color?: PropertyColor;
}
